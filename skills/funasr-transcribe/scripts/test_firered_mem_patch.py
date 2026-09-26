# 自测: firered_mem_patch 分块注意力数值等价性 + qwen_asr AED 显存兜底纯逻辑 + GPU 显存对比
# 运行: python scripts/test_firered_mem_patch.py
# 依赖: fire-red-asr2s 克隆(~/.local/fire-red-asr2s)与 torch;无 GPU 时显存对比段自动跳过
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        fails.append(name)


AED_CODE_DIR = Path.home() / ".local" / "fire-red-asr2s"
if not (AED_CODE_DIR / "fireredasr2s").is_dir():
    sys.exit(f"FireRedASR2S 代码目录不存在: {AED_CODE_DIR}(克隆: git clone "
             f"https://github.com/FireRedTeam/FireRedASR2S.git \"{AED_CODE_DIR}\")")
sys.path.insert(0, str(AED_CODE_DIR))

from fireredasr2s.fireredasr2.models.module import conformer_encoder as ce
from fireredasr2s.fireredasr2.models.module import transformer_decoder as td
import firered_mem_patch

# 补丁前快照原始实现(等价性基准)
orig_relpos = ce.RelPosMultiHeadAttention.forward
orig_dec = td.DecoderScaledDotProductAttention.forward

print("[1] RelPosMultiHeadAttention 等价性(带 mask,块边界不整除)")
torch.manual_seed(0)
attn_mod = ce.RelPosMultiHeadAttention(n_head=4, d_model=64, residual_dropout=0.0).eval()
B, T, D = 2, 50, 64
x = torch.randn(B, T, D)
pos_emb = ce.RelPositionalEncoding(D)(x)
mask = torch.ones(B, 1, T, dtype=torch.uint8)
mask[1, 0, 30:] = 0  # 第二条后 20 帧为 padding
with torch.no_grad():
    ref_out, _ = orig_relpos(attn_mod, x, x, x, pos_emb, mask=mask)
firered_mem_patch.apply(chunk=7)  # 50 帧 / 块 7 → 余 1,逼出块边界
with torch.no_grad():
    new_out, new_attn = attn_mod(x, x, x, pos_emb, mask=mask)
diff = (ref_out - new_out).abs().max().item()
check("输出逐元素等价", torch.allclose(ref_out, new_out, atol=1e-6), f"max diff {diff:.3e}")
check("attn 权重返回 None(调用点均丢弃)", new_attn is None)

print("[2] _rel_shift_chunk 切片重组 vs 原版整块 cat(等价性)")
m_shift = ce.RelPosMultiHeadAttention(n_head=4, d_model=64, residual_dropout=0.0)
raw = torch.randn(2, 4, 30, 59)  # T=30, 2T-1=59
ref_shift = m_shift._rel_shift(raw)
pieces = []
for s in range(0, 30, 7):  # 块边界不整除
    e = min(s + 7, 30)
    pieces.append(firered_mem_patch._rel_shift_chunk(raw[:, :, s:e], s, 30))
new_shift = torch.cat(pieces, dim=2)
diff = (ref_shift - new_shift).abs().max().item()
check("rel_shift 分块逐元素等价", torch.allclose(ref_shift, new_shift, atol=1e-6),
      f"max diff {diff:.3e}")

print("[3] DecoderScaledDotProductAttention 等价性(4D 因果 mask / 3D 交叉 mask / None)")
dec = td.DecoderScaledDotProductAttention(temperature=4)
q = torch.randn(2, 4, 17, 16)
k_self = torch.randn(2, 4, 17, 16)   # 自注意力 q/k 等长
v_self = torch.randn(2, 4, 17, 16)
k_cross = torch.randn(2, 4, 23, 16)  # 交叉注意力 k=编码器帧数
v_cross = torch.randn(2, 4, 23, 16)
mask4 = torch.tril(torch.ones(17, 17))[None, None].to(torch.uint8).repeat(2, 1, 1, 1)
mask3 = torch.ones(2, 1, 1, 23, dtype=torch.uint8)  # 交叉 mask 经 DecoderMHA unsqueeze 后恒 4 维
mask3[1, 0, 0, 10:] = 0
for tag, kk, vv, m in (("4D 自注意力", k_self, v_self, mask4),
                       ("4D 交叉注意力", k_cross, v_cross, mask3),
                       ("None", k_cross, v_cross, None)):
    with torch.no_grad():
        ref = orig_dec(dec, q, kk, vv, mask=m)
        new = dec(q, kk, vv, mask=m)
    diff = (ref - new).abs().max().item()
    check(f"decoder {tag} 等价", torch.allclose(ref, new, atol=1e-6), f"max diff {diff:.3e}")

print("[3b] upper_triangular_is_0 缓存(数值等价 + 命中同对象 + 覆盖 restore/apply 往返)")
m_td = td.TransformerDecoder(sos_id=1, eos_id=2, pad_id=0, odim=10, n_layers=1,
                             d_model=16, n_head=2)
ref_tri = firered_mem_patch._originals["upper_tri"](m_td, 9)   # [1] 已 apply,存的是原始实现
new_tri = m_td.upper_triangular_is_0(9)
check("缓存值与原始实现逐元素一致", torch.equal(ref_tri, new_tri.cpu()),
      f"dtype {ref_tri.dtype} vs {new_tri.dtype}")
check("二次调用命中缓存(同对象)", m_td.upper_triangular_is_0(9) is new_tri)
check("缓存张量在 decoder 设备上", new_tri.device == m_td.tgt_word_prj.weight.device)

print("[4] qwen_asr.aed_batch_plan 时长感知分批")
from qwen_asr import (aed_batch_plan, aed_transcribe_oom_safe, merge_aed_results,
                      split_wav_at_quiet)
import soundfile as sf

tmp = Path(tempfile.mkdtemp(prefix="t_mem_patch_"))
wavs = []
for i, dur in enumerate((2.0, 3.0, 4.0)):
    p = tmp / f"w{i}.wav"
    sf.write(str(p), np.zeros(int(16000 * dur), dtype=np.float32), 16000)
    wavs.append(str(p))
plan = aed_batch_plan(wavs, batch_size=8, budget_s=5.0)
check("时长预算分批 2+3/4", plan == [wavs[:2], wavs[2:]], f"got {plan}")
plan = aed_batch_plan(wavs, batch_size=2, budget_s=100.0)
check("批大小上限分批", [len(b) for b in plan] == [2, 1], f"got {plan}")
plan = aed_batch_plan(wavs, batch_size=8, budget_s=0.0)
check("预算 0 仍受批大小限制", len(plan) == 3 and all(len(b) == 1 for b in plan), f"got {plan}")
plan = aed_batch_plan([], 8, 120.0)
check("空输入返回空计划", plan == [])

print("[5] split_wav_at_quiet 静音点二分")
sr = 16000
y = np.sin(np.linspace(0, 400 * 2 * np.pi, int(2.0 * sr))).astype(np.float32) * 0.5
y[int(1.0 * sr):int(1.2 * sr)] = 0.0  # 1.0-1.2s 静音
p = tmp / "seg.wav"
sf.write(str(p), y, sr)
pa, pb, split_ms = split_wav_at_quiet(p, tmp, 0)
check("切点落在静音区", 0.98 <= split_ms / 1000.0 <= 1.22, f"got {split_ms / 1000.0:.3f}s")
check("前后段文件生成", os.path.isfile(pa) and os.path.isfile(pb))
da = sf.info(pa).duration + sf.info(pb).duration
check("前后段时长守恒", abs(da - 2.0) < 0.01, f"got {da:.3f}s")

print("[6] merge_aed_results 拼接/平移/置信度")
ra = [{"uttid": "u0", "text": "你好世界", "confidence": 0.9, "dur_s": 1.0,
       "timestamp": [["你", 0.1, 0.2], ["好", 0.2, 0.3]]}]
rb = [{"uttid": "a", "text": "hello", "confidence": 0.7, "dur_s": 0.5,
       "timestamp": [["h", 0.1, 0.15]]}]
out = merge_aed_results(ra, rb, offset_s=2.0, uttid="u0")
check("单元素返回", len(out) == 1 and out[0]["uttid"] == "u0")
check("中英边界补空格", out[0]["text"] == "你好世界 hello", f"got {out[0]['text']!r}")
check("时间戳平移", out[0]["timestamp"] == [["你", 0.1, 0.2], ["好", 0.2, 0.3], ["h", 2.1, 2.15]],
      f"got {out[0]['timestamp']}")
check("置信度取小", out[0]["confidence"] == 0.7)
check("时长求和", out[0]["dur_s"] == 1.5)
out2 = merge_aed_results([{"uttid": "a", "text": "世界。"}], [{"uttid": "b", "text": "你好"}], 1.0, "u0")
check("中文边界不补空格", out2[0]["text"] == "世界。你好", f"got {out2[0]['text']!r}")

print("[7] aed_transcribe_oom_safe 重试链(FakeModel)")


class FakeModel:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    def transcribe(self, uttids, paths):
        self.calls.append(list(paths))
        return self.behavior(uttids, paths)


def ok(u, paths):
    return [{"uttid": i, "text": f"ok:{Path(p).stem}", "confidence": 0.9,
             "dur_s": sf.info(p).duration, "timestamp": [["你", 0.0, 0.5]]}
            for i, p in zip(u, paths)]


def empty(u, paths):
    return [{"uttid": i, "text": ""} for i in u]


logs = []
ctx = {"work_dir": tmp, "log": logs.append, "min_split_s": 0.5, "max_depth": 2}

# 用例 A: 批量空 → 拆单全成功,顺序保持
beh_a = lambda u, paths: empty(u, paths) if len(paths) > 1 else ok(u, paths)
m = FakeModel(beh_a)
res = aed_transcribe_oom_safe(m, wavs, ctx)
check("拆单后全成功", [r["text"] for r in res] == [f"ok:w{i}" for i in range(3)], f"got {[r['text'] for r in res]}")
check("调用序列 1 批 + 3 单", len(m.calls) == 4 and len(m.calls[0]) == 3, f"got {len(m.calls)}")

# 用例 B: 单长段空 → 二分重试并合并
wlong = tmp / "long.wav"
sf.write(str(wlong), y, sr)  # 2s 带静音
beh_b = lambda u, paths: ok(u, paths) if any("_d" in p for p in paths) else empty(u, paths)
m = FakeModel(beh_b)
res = aed_transcribe_oom_safe(m, [str(wlong)], ctx)
check("二分重试合并为单结果", len(res) == 1, f"got {len(res)}")
check("合并文本含两半", " " in res[0]["text"] and res[0]["text"].startswith("ok:long"), f"got {res[0]['text']!r}")
check("合并含时间戳平移", len(res[0]["timestamp"]) == 2 and res[0]["timestamp"][1][1] > 0.5,
      f"got {res[0]['timestamp']}")
check("重试链有日志", any("拆小重试" in s for s in logs))

# 用例 C: 段短于 min_split_s → 不重试
m = FakeModel(empty)
res = aed_transcribe_oom_safe(m, [wavs[0]], {**ctx, "min_split_s": 100.0})
check("短段不重试直接返回", len(m.calls) == 1 and res[0]["text"] == "")

# 用例 D: 多短段批空 → 拆单后各自因过短放弃,结果仍按序补齐
m = FakeModel(empty)
res = aed_transcribe_oom_safe(m, wavs, {**ctx, "min_split_s": 100.0})
check("拆单到限返回原状", len(m.calls) == 4 and [r["text"] for r in res] == [""] * 3,
      f"calls={len(m.calls)}")

print("[8] GPU 显存对比(patch 前 vs 后)" + ("" if torch.cuda.is_available() else " —— 无 GPU,跳过"))
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m_gpu = ce.RelPosMultiHeadAttention(n_head=8, d_model=512, residual_dropout=0.0).cuda().eval()
    xg = torch.randn(2, 1200, 512, device="cuda")
    posg = ce.RelPositionalEncoding(512).cuda()(xg)
    mg = torch.ones(2, 1, 1200, dtype=torch.uint8, device="cuda")
    with torch.no_grad():
        firered_mem_patch.restore()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        orig_relpos(m_gpu, xg, xg, xg, posg, mask=mg)
        peak_orig = torch.cuda.max_memory_allocated() / 2**20
        firered_mem_patch.apply()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m_gpu(xg, xg, xg, posg, mask=mg)
        peak_new = torch.cuda.max_memory_allocated() / 2**20
    check(f"patched 峰值更低({peak_orig:.0f}MB -> {peak_new:.0f}MB)", peak_new < peak_orig * 0.8,
          f"orig {peak_orig:.0f}MB new {peak_new:.0f}MB")

print()
if fails:
    print(f"FAIL {len(fails)} 项: {fails}")
    sys.exit(1)
print("ALL PASS")
