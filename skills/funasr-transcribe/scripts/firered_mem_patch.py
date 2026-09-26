# FireRedASR2 显存优化补丁: 分块注意力(monkey-patch,不改 clone 源码)
# 运行: 由 qwen_asr.py load_aed_model() 调 apply();自测: python scripts/test_firered_mem_patch.py
#
# 背景: FireRedASR2-AED 的注意力原实现整块物化 (B,H,T,T) 分数矩阵并在 softmax
#       前后多次非原地拷贝;相对位置路径的 matrix_bd 还要过 _rel_shift 的 cat
#       (再翻一倍)。实测 4×107s 批量(B=4,H=20,T≈2680)仅 bd 中间量即 ~8.6GB,
#       单个 cat 分配 4.28GiB 直接 OOM(FireRed 内部吞异常,表现为整批空文本)。
# 原理: 沿 query 维分块现算现用——softmax 按行独立,分块与整块数学严格等价;
#       相对位置偏置按行切片重组(bd[i,j] = raw[i, j-i+T-1]),单次分配无 cat;
#       ac/bd 块内就地累加。注意力峰值从 O(B·H·T²) 降到 O(chunk·B·H·T)。
# 范围: conformer_encoder.RelPosMultiHeadAttention.forward(融合分块主路径)+
#       transformer_decoder.DecoderScaledDotProductAttention.forward(逐步自/
#       交叉注意力)。编码器普通路径 EncoderMultiHeadAttention.forward 实际无
#       实例(RelPos 子类独占),不补。注意力权重原实现即被丢弃(调用点取 [0])。
# bf16 配套(use_half=True): ① TransformerDecoder.upper_triangular_is_0 按
#       (size,device) 缓存——原实现每个 beam step 在 CPU 重建下三角再传 GPU;
#       ② CTC.forward 输出 cast fp32——torchaudio forced_align 内核要求 float32,
#       bf16 权重下 ctc_logits 为 bf16,直喂会抛异常并被 asr.py 吞掉,字级时间戳
#       退化为均匀估计(ctc 模块内部 matmul 仍走 bf16 同型,cast 只落在输出)。
#       fp32 下均为 no-op。

import torch

_QUERY_CHUNK = 256
_originals = {}
_applied = False


def _rel_shift_chunk(raw_c, s, T):
    """等价于原 _rel_shift 的整块变换,按行切片重组(单次分配,无 cat 大拷贝)。
    原变换语义: out[i, j] = raw[i, j - i + T - 1];raw_c 为 query 块 [s, s+c) 的
    q·p^T 分数 (N,H,c,2T-1),返回 (N,H,c,T)。切点范围恒在 [0, 2T-1) 内(c<=T)。"""
    delta = T - 1 - s
    rows = [raw_c[:, :, r, delta - r: delta - r + T] for r in range(raw_c.size(2))]
    return torch.stack(rows, dim=2)


def _patched_relpos_forward(self, q, k, v, pos_emb, mask=None):
    """与原 RelPosMultiHeadAttention.forward 数学严格一致,按 query 块融合:
    块内算 matrix_ac/matrix_bd(切片重组替代 _rel_shift 的 cat)、就地累加缩放、
    softmax/mask/dropout/×v,峰值从 O(T²) 降到 O(chunk·T)。"""
    sz_b, len_q = q.size(0), q.size(1)

    residual = q
    q, k, v = self.forward_qkv(q, k, v)          # 各 (N,H,T,d)

    q = q.transpose(1, 2)                         # (N,T,H,d)
    n_batch_pos = pos_emb.size(0)
    p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
    p = p.transpose(1, 2)                         # (1,H,2T-1,d)

    q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)   # (N,H,T,d)
    q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

    T = k.size(2)
    pad = mask.unsqueeze(1).eq(0) if mask is not None else None  # (N,1,1,T)
    kT = k.transpose(-2, -1)
    pT = p.transpose(-2, -1)
    inf, dropout = self.attention.INF, self.attention.dropout
    outs = []
    for s in range(0, len_q, _QUERY_CHUNK):
        e = min(s + _QUERY_CHUNK, len_q)
        ac = torch.matmul(q_with_bias_u[:, :, s:e], kT)          # (N,H,c,T)
        raw = torch.matmul(q_with_bias_v[:, :, s:e], pT)         # (N,H,c,2T-1)
        bd = _rel_shift_chunk(raw, s, T)
        del raw
        scores = ac.add_(bd).mul_(self.scale)                    # 就地累加+缩放
        del bd
        if pad is not None:
            scores = scores.masked_fill(pad, -inf)
            attn = torch.softmax(scores, dim=-1).masked_fill(pad, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        attn = dropout(attn)
        outs.append(torch.matmul(attn, v))
    output = torch.cat(outs, dim=2)

    output = self.forward_output(output, residual, sz_b, len_q)
    return output, None


def _patched_decoder_forward(self, q, k, v, mask=None):
    """解码器逐步自/交叉注意力: 分数按 query 块现算现用,不再整块物化 (NB,H,t,T)。
    mask 已由 DecoderMultiHeadAttention unsqueeze,恒为 4 维: 自注意力 (NB,1,t,t)
    (query 维真实长度,需按块同步切) / 交叉 (NB,1,1,Ti) (query 维为广播单例,不切)"""
    pad = mask.eq(0) if mask is not None else None
    outs = []
    for s in range(0, q.size(2), _QUERY_CHUNK):
        qc = q[:, :, s:s + _QUERY_CHUNK]
        a = torch.matmul(qc, k.transpose(2, 3)) / self.temperature
        p = pad[:, :, s:s + qc.size(2)] if (pad is not None and pad.dim() == 4
                                            and pad.size(2) > 1) else pad
        if p is not None:
            a = a.masked_fill(p, -self.INF)
            a = torch.softmax(a, dim=-1).masked_fill(p, 0.0)
        else:
            a = torch.softmax(a, dim=-1)
        outs.append(torch.matmul(a, v))
    return torch.cat(outs, dim=2)


_tri_cache = {}


def _patched_upper_triangular_is_0(self, size):
    """三角 mask 按 (size,device) 缓存:原实现每个 beam step 在 CPU 重建 size×size
    下三角(uint8)再传 GPU,输出几百步时重建+传输累计可观。数值与原实现逐元素
    一致;调用处仅做 .to(dtype) 与 & 运算,不会就地修改缓存张量。"""
    dev = self.tgt_word_prj.weight.device
    key = (size, dev.type, dev.index)
    t = _tri_cache.get(key)
    if t is None:
        t = torch.tril(torch.ones(size, size)).to(torch.uint8).to(dev)
        _tri_cache[key] = t
    return t


def _patched_ctc_forward(self, hid):
    """CTC 输出 cast fp32:torchaudio forced_align 内核要求 float32,bf16 权重下
    ctc_logits 为 bf16,直喂会抛异常并被 asr.py 吞掉,字级时间戳退化为均匀估计。
    模块内部 matmul 保持 bf16 同型(输入 cast 反而 dtype 不匹配);fp32 下 no-op。
    推理期 CTC 唯一调用点是 get_token_timestamp_torchaudio,改输出 dtype 无副作用。"""
    out = _originals["ctc_forward"](self, hid)
    return out.float() if out.dtype != torch.float32 else out


def apply(chunk=256):
    """给 FireRedASR2 类挂分块实现(幂等);须在 sys.path 含 fire-red-asr2s 克隆后调用。
    返回补丁说明字符串供日志。"""
    global _QUERY_CHUNK, _applied
    _QUERY_CHUNK = max(32, int(chunk))
    from fireredasr2s.fireredasr2.models.module import conformer_encoder as ce
    from fireredasr2s.fireredasr2.models.module import transformer_decoder as td
    from fireredasr2s.fireredasr2.models.module import ctc as ctc_mod
    if not _applied:
        _originals.update(
            relpos_forward=ce.RelPosMultiHeadAttention.forward,
            dec_attn=td.DecoderScaledDotProductAttention.forward,
            upper_tri=td.TransformerDecoder.upper_triangular_is_0,
            ctc_forward=ctc_mod.CTC.forward,
        )
        ce.RelPosMultiHeadAttention.forward = _patched_relpos_forward
        td.DecoderScaledDotProductAttention.forward = _patched_decoder_forward
        td.TransformerDecoder.upper_triangular_is_0 = _patched_upper_triangular_is_0
        ctc_mod.CTC.forward = _patched_ctc_forward
        _applied = True
    return (f"firered 分块注意力补丁已生效(query 块={_QUERY_CHUNK}, "
            f"编码器 RelPos 融合分块+解码器逐步注意力+三角 mask 缓存+CTC 输出 fp32)")


def restore():
    """还原原始实现(测试/排障用)"""
    global _applied
    if not _originals:
        return
    from fireredasr2s.fireredasr2.models.module import conformer_encoder as ce
    from fireredasr2s.fireredasr2.models.module import transformer_decoder as td
    from fireredasr2s.fireredasr2.models.module import ctc as ctc_mod
    ce.RelPosMultiHeadAttention.forward = _originals["relpos_forward"]
    td.DecoderScaledDotProductAttention.forward = _originals["dec_attn"]
    td.TransformerDecoder.upper_triangular_is_0 = _originals["upper_tri"]
    ctc_mod.CTC.forward = _originals["ctc_forward"]
    _applied = False
