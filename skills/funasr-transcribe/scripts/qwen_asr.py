# 本地转写引擎（双引擎：Qwen3-ASR-1.7B 默认 / FireRedASR2-AED 可选）
# 混搭方案：识别引擎（Qwen3-ASR 或 FireRedASR2-AED）+ FunASR fsmn-vad/CAM++ 说话人分离
#           + 字级时间戳/SRT + 专名确定性纠错 + FireRedVAD 可选 VAD + ZipEnhancer 可选降噪
# 与技能 funasr-transcribe 的关系：识别引擎从 Fun-ASR-Nano 升级为 Qwen3-ASR-1.7B；
# 分环逻辑（VAD 切段 -> 逐段声纹 -> 余弦层次聚类）与技能 diarize.py 完全一致。
#
# 对接真相表（证据 = 本机源码/实测）:
#   构造输入  processor.apply_transcription_request(audio, language, prompt, return_tensors="pt")
#             -> processing_qwen3_asr.py:494 (chat template + assistant prefill "language <NAME><asr_text>")
#   生成      Qwen3ASRForConditionalGeneration.generate(**inputs) -> modeling_qwen3_asr.py 头部 Example
#   解析      processor.decode(ids, return_format="parsed") -> {"language","transcription"}
#   音频输入  本地路径/URL/numpy，或其列表（批量）；路径模式由 processor 内部读取重采样
#   模型仓库  Qwen/Qwen3-ASR-1.7B-hf（ModelScope, -hf 后缀 = transformers 原生格式）, 4.08GB
#   备选引擎  FireRedASR2-AED（ModelScope `xukaituo/FireRedASR2-AED`, 4.73GB; 代码 https://github.com/FireRedTeam/FireRedASR2S）
#             FireRedAsr2.from_pretrained("aed", model_dir, cfg).transcribe(uttids, wav_paths)
#             -> [{'uttid','text','confidence','timestamp':[('字',start_s,end_s),...]}]  (字级时间戳原生)
#             输入限制: kaldiio.load_mat 只认 16k 单声道 wav（mp3 会报 read_ascii_mat 错）→ 脚本内自动转
#             实测: 中/英/粤/噪声音频优于 Qwen3-ASR；日文不支持（输出中文乱码）；
#                   输出无标点 → 可选 ct-punc 补标点（--no-punc 关闭）；首调约 7-8s 预热,之后 2.7ms/秒音频
#   强制对齐  funasr AutoModel("fa-zh") = iic/speech_timestamp_prediction-v1-16k-offline (38M)
#             调用 input=(wav, text) / ([wavs],[texts])，data_type=("sound","text")
#             输出 {"text": "字 字 字"(空格分隔), "timestamp": [[start_ms,end_ms],...]}
#             实测: 标点自动跳过（13 汉字+标点输入 -> 13 token/13 时间戳）；66s 单次对齐 OK
#   专名纠错  funasr.utils.postprocess_hotwords.build_postprocess_hotword_matcher
#             确定性 dict{错:对} 替换；--fuzzy 走拼音模糊（需 pypinyin+rapidfuzz）
#   VAD 选择  fsmn-vad（默认，随 funasr）/ fireredvad（pip 包 + xukaituo/FireRedVAD 缓存）
#
# 路径安全边界: 输入拒绝 ".." 且 realpath 规范化；输出 JSON/SRT 锁定音频同目录；
#               临时分段 wav 锁定系统临时目录；落盘前均显式校验目录边界。
#
# 用法:
#   python qwen_asr.py 会议录音.m4a                     # 整段转写（自动语言检测）
#   python qwen_asr.py 会议录音.m4a --srt               # 额外输出 .srt 字幕（字级时间戳）
#   python qwen_asr.py 会议录音.m4a --replace "开饭时间=>开放时间"   # 专名确定性纠错
#   python qwen_asr.py 采访.m4a --diarize --names "0=张三,1=李四"   # 说话人分离 + 重命名
#   python qwen_asr.py 访谈.m4a --diarize --threshold 0.7           # 聚类阈值调节
#   python qwen_asr.py 录音.m4a --engine aed            # 换 FireRedASR2-AED（中文更准；日文不支持）
#   python qwen_asr.py 录音.m4a --vad firered            # 换 FireRedVAD 切段
#   python qwen_asr.py 嘈杂.m4a --denoise                # ZipEnhancer 前处理（含 BGM/强噪时试）
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO_ID = "Qwen/Qwen3-ASR-1.7B-hf"
MS_CACHE = Path(os.path.expanduser("~")) / ".cache" / "modelscope" / "models" / "Qwen--Qwen3-ASR-1.7B-hf" / "snapshots" / "master"
AED_MODEL_DIR = Path.home() / ".cache" / "modelscope" / "models" / "xukaituo--FireRedASR2-AED"
AED_CODE_DIR = Path.home() / ".local" / "fire-red-asr2s"
FIRERED_VAD_DIR = Path(os.path.expanduser("~")) / ".cache" / "fireredvad" / "FireRedVAD" / "VAD"
AUTO_SEGMENT_S = 60  # 超过此时长自动走 VAD 分段批量转写(实测比整段自回归快约 20%,且每段语言独立检测)
PUNCT = set("，。！？；：、,.!?;:…—～~「」『』“”‘’（）()《》〈〉【】[]\"'· \t\n\r")
HARD_STOP = "。！？；!?;"
SOFT_STOP = "，,、：:"


def resolve_model_path():
    return str(MS_CACHE) if MS_CACHE.is_dir() else REPO_ID


def load_model(device):
    """引擎 1(默认): Qwen3-ASR-1.7B,transformers 原生直载"""
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(resolve_model_path())
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        resolve_model_path(), dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    return model, processor, time.time() - t0


def load_aed_model(device):
    """引擎 2: FireRedASR2-AED（中文/英文/粤语实测优于 Qwen3-ASR；不支持日文；输出无标点）"""
    import sys
    if not (AED_MODEL_DIR / "model.pth.tar").is_file():
        sys.exit(f"FireRedASR2-AED 模型不存在: {AED_MODEL_DIR}\n"
                 f"下载: python -c \"from modelscope import snapshot_download; "
                 f"snapshot_download('xukaituo/FireRedASR2-AED', local_dir=r'{AED_MODEL_DIR}')\"")
    if not (AED_CODE_DIR / "fireredasr2s").is_dir():
        sys.exit(f"FireRedASR2S 代码目录不存在: {AED_CODE_DIR}\n"
                 f"克隆: git clone https://github.com/FireRedTeam/FireRedASR2S.git \"{AED_CODE_DIR}\"")
    if str(AED_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(AED_CODE_DIR))
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
    t0 = time.time()
    cfg = FireRedAsr2Config(
        use_gpu=str(device).startswith("cuda"), use_half=False, beam_size=3, nbest=1,
        decode_max_len=0, softmax_smoothing=1.25, aed_length_penalty=0.6, eos_penalty=1.0,
        return_timestamp=True,
    )
    model = FireRedAsr2.from_pretrained("aed", str(AED_MODEL_DIR), cfg)
    return model, None, time.time() - t0


def load_punc(device):
    """ct-punc 标点恢复(AED 引擎输出无标点,用它补;加载失败返回 None 不阻断)"""
    try:
        from funasr import AutoModel
        return AutoModel(model="ct-punc", device=device, disable_update=True)
    except Exception as e:
        print(f"[warn] ct-punc 加载失败,跳过补标点: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None


def transcribe_batch(model, processor, audio_paths, language=None, hotwords=None, max_new_tokens=2048,
                     t2s=None, matcher=None, punc=None, engine="qwen"):
    """批量转写,返回 [{"language","transcription"}, ...]
    Qwen 引擎: 支持 language/hotwords 偏置、自带标点与语言识别
    AED 引擎: 忽略 language/hotwords（模型无此接口）；输出无标点,经 ct-punc 补；时间戳由 AED 原生提供"""
    if engine == "aed":
        uttids = [f"u{i}" for i in range(len(audio_paths))]
        results = model.transcribe(uttids, list(audio_paths))
        texts = [(r.get("text") or "").strip() for r in results]
        # ct-punc 批量调用(逐段调用实测 111 段多花 17s;批量后降到亚秒级)
        if punc is not None and any(texts):
            try:
                punc_out = punc.generate(input=[t if t else "。" for t in texts])
                texts = [(p.get("text") or t).strip() for p, t in zip(punc_out, texts)]
            except Exception:
                pass
        parsed = []
        for r, txt in zip(results, texts):
            if t2s is not None:
                txt = t2s(txt)
            parsed.append({"language": None, "transcription": apply_replace(txt, matcher),
                           "timestamp": r.get("timestamp"), "confidence": r.get("confidence")})
        return parsed
    import torch
    inputs = processor.apply_transcription_request(
        audio=audio_paths, language=language, prompt=hotwords, return_tensors="pt")
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    # 音频特征 extractor 输出 float32,与 bf16 模型权重对齐(实测 conv 层要求同型)
    if "input_features" in inputs and torch.is_tensor(inputs["input_features"]):
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    with torch.inference_mode():
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    out_ids = gen[:, inputs["input_ids"].shape[1]:]
    parsed = processor.decode(out_ids, return_format="parsed")
    for item in parsed:
        if t2s is not None and item.get("language") in ("Chinese", "Cantonese"):
            item["transcription"] = t2s(item["transcription"])
        item["transcription"] = apply_replace(item["transcription"], matcher)
    return parsed


def build_t2s(disabled):
    """繁->简转换器:默认启用,仅对中文/粤语输出生效(日文等语言汉字不误转)"""
    if disabled:
        return None
    from opencc import OpenCC
    return OpenCC("t2s").convert


def build_matcher(replace_map, fuzzy):
    """专名纠错 matcher:确定性替换(dict) + 可选拼音模糊;返回可 apply_text 的对象或 None"""
    if not replace_map:
        return None
    from funasr.utils.postprocess_hotwords import build_postprocess_hotword_matcher
    return build_postprocess_hotword_matcher(
        postprocess_hotwords=replace_map, enable_fuzzy=bool(fuzzy))


def apply_replace(text, matcher):
    if matcher is None:
        return text
    return matcher.apply_text(text)[0]


# ---------------- VAD / 对齐 / 字幕 ----------------

def run_vad(audio_path, backend, device):
    """返回 [[start_ms, end_ms], ...]；fsmn 走 funasr，firered 走 fireredvad"""
    if backend == "firered":
        from fireredvad import FireRedVad, FireRedVadConfig
        if not (FIRERED_VAD_DIR / "model.pth.tar").is_file():
            sys.exit(f"FireRedVAD 模型不存在: {FIRERED_VAD_DIR}（需先 modelscope download --model xukaituo/FireRedVAD）")
        cfg = FireRedVadConfig(use_gpu=str(device).startswith("cuda"))
        vad = FireRedVad.from_pretrained(str(FIRERED_VAD_DIR), cfg)
        result, _ = vad.detect(str(audio_path))
        return [[int(round(s * 1000)), int(round(e * 1000))] for s, e in result["timestamps"]]
    from funasr import AutoModel
    vad = AutoModel(model="fsmn-vad", device=device, disable_update=True)
    return [[int(s), int(e)] for s, e in vad.generate(input=str(audio_path))[0]["value"]]


def aed_aligns_from(parsed):
    """AED 原生字级时间戳(秒,含字符) -> 与 fa-zh 同构的 aligns 结构(毫秒)
    注意: AED 时间戳对应的是补标点前的字符序列;补标点后文本多了标点,
          char_times/subtitle_units 会按去标点字符数比对,数量一致即按序映射"""
    out = []
    for p in parsed:
        ts = p.get("timestamp") or []
        out.append({"tokens": [t[0] for t in ts],
                    "ts": [[int(round(t[1] * 1000)), int(round(t[2] * 1000))] for t in ts]})
    return out


def aed_ts_usable(parsed):
    """AED 原生时间戳是否可用: 时间戳数 == 文本去标点字符数(补标点致长度不等是正常的)"""
    for p in parsed:
        ts = p.get("timestamp") or []
        text = p.get("transcription") or ""
        if not ts:
            return False
        if len(ts) != sum(1 for c in text if c not in PUNCT):
            return False
    return True


def align_batch(fa_model, wav_paths, texts):
    """fa-zh 强制对齐;返回 [{"tokens":[...],"ts":[[s,e],...]}, ...],顺序与输入一致"""
    out = []
    for i in range(0, len(wav_paths), 8):
        chunk_a, chunk_t = wav_paths[i:i + 8], texts[i:i + 8]
        res = fa_model.generate(input=(chunk_a, chunk_t), data_type=("sound", "text"))
        for item in res:
            out.append({"tokens": item["text"].split(), "ts": item["timestamp"]})
    return out


def char_times(text, tokens, ts, base_ms, fallback_start, fallback_end):
    """去标点字符序 -> [(start_ms,end_ms)];token 数与字符数一致时按序映射,否则按跨度均分兜底"""
    plain_len = sum(1 for c in text if c not in PUNCT)
    if plain_len <= 0:
        return []
    if len(ts) == plain_len:
        out, k = [], 0
        for c in text:
            if c in PUNCT:
                continue
            out.append((int(ts[k][0]) + base_ms, int(ts[k][1]) + base_ms))
            k += 1
        return out
    # 兜底:在给定跨度内均分(对齐 token 与文本不匹配时,保时间轴合理)
    span = max(1, fallback_end - fallback_start)
    return [(fallback_start + span * i // plain_len, fallback_start + span * (i + 1) // plain_len)
            for i in range(plain_len)]


def split_sentences(text, max_len):
    """按标点切句,超长句再按软标点细分;返回 [(plain_start, plain_end, sentence)]
    plain 索引 = 去标点后的字符序(与 char_times 的键一致)"""
    # 字符位置 -> plain 索引(标点处为 None)
    plain_of, pi = [], 0
    for ch in text:
        if ch in PUNCT:
            plain_of.append(None)
        else:
            plain_of.append(pi)
            pi += 1
    # 硬标点切句(字符区间)
    spans, start = [], 0
    for i, ch in enumerate(text):
        if ch in HARD_STOP:
            if text[start:i + 1].strip():
                spans.append((start, i + 1))
            start = i + 1
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    # 超长句按软标点细分
    pieces = []
    for cs, ce in spans:
        if len(text[cs:ce]) <= max_len:
            pieces.append((cs, ce))
            continue
        sub_start = cs
        for i in range(cs, ce):
            if text[i] in SOFT_STOP and (i - sub_start) >= max_len // 2:
                pieces.append((sub_start, i + 1))
                sub_start = i + 1
        if sub_start < ce:
            pieces.append((sub_start, ce))
    # 转 plain 区间
    res = []
    for cs, ce in pieces:
        pa = next((plain_of[i] for i in range(cs, ce) if plain_of[i] is not None), None)
        pb = next((plain_of[i] for i in reversed(range(cs, ce)) if plain_of[i] is not None), None)
        sent = text[cs:ce].strip()
        if pa is not None and pb is not None and sent:
            res.append((pa, pb + 1, sent))
    return res


def subtitle_units(units, aligns, max_len):
    """把输出单元细分为字幕条;units=[{start_ms,end_ms,text,spk,spk_name}],aligns 可为 None
    有 aligns 时按字级时间戳细分并校正边界,无则整段一条"""
    subs = []
    for idx, u in enumerate(units):
        al = aligns[idx] if aligns else None
        if not al or not al.get("ts"):
            subs.append({"start_ms": u["start_ms"], "end_ms": u["end_ms"], "text": u["text"],
                         "spk": u.get("spk"), "spk_name": u.get("spk_name")})
            continue
        ct = char_times(u["text"], al["tokens"], al["ts"], u["start_ms"], u["start_ms"], u["end_ms"])
        for a, b, sent in split_sentences(u["text"], max_len):
            if not ct:
                seg_s, seg_e = u["start_ms"], u["end_ms"]
            else:
                seg_s = ct[min(a, len(ct) - 1)][0]
                seg_e = ct[min(max(b - 1, 0), len(ct) - 1)][1]
            subs.append({"start_ms": seg_s, "end_ms": seg_e, "text": sent,
                         "spk": u.get("spk"), "spk_name": u.get("spk_name")})
    # 相邻字幕时间收敛:上一条 end 不晚于下一条 start
    for i in range(len(subs) - 1):
        if subs[i]["end_ms"] > subs[i + 1]["start_ms"]:
            subs[i]["end_ms"] = subs[i + 1]["start_ms"]
        if subs[i]["end_ms"] <= subs[i]["start_ms"]:
            subs[i]["end_ms"] = subs[i]["start_ms"] + 200
    return subs


def fmt_srt_ts(ms):
    ms = max(0, int(ms))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def write_srt(subs, path, use_speaker):
    blocks = []
    for i, s in enumerate(subs, 1):
        label = ""
        if use_speaker and (s.get("spk_name") or s.get("spk") is not None):
            label = f"[{s.get('spk_name') or ('说话人' + str(s['spk']))}] "
        blocks.append(f"{i}\n{fmt_srt_ts(s['start_ms'])} --> {fmt_srt_ts(s['end_ms'])}\n{label}{s['text']}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def merge_by_speaker(segments, labels, gap_ms):
    """相邻同说话人且间隔 <= gap_ms 的段合并;返回 [{"start_ms","end_ms","spk","seg_idx"}]"""
    merged = []
    for i, (s, e) in enumerate(segments):
        s, e = int(s), int(e)
        spk = int(labels[i])
        if merged and merged[-1]["spk"] == spk and s - merged[-1]["end_ms"] <= gap_ms:
            merged[-1]["end_ms"] = e
            merged[-1]["seg_idx"].append(i)
        else:
            merged.append({"start_ms": s, "end_ms": e, "spk": spk, "seg_idx": [i]})
    return merged


def write_unit_wavs(audio, units, tmp_dir):
    """按单元切临时 wav(边界显式校验)"""
    import soundfile as sf
    files = []
    for i, u in enumerate(units):
        p = tmp_dir / f"unit_{i:04d}.wav"
        if p.parent != tmp_dir:
            raise RuntimeError("临时分段路径越出临时目录,已终止")
        sf.write(str(p), audio[u["start_ms"] * 16: u["end_ms"] * 16], 16000)
        files.append(str(p))
    return files


def to_wav16k(audio_path, tmp_dir):
    """转 16k 单声道 wav（AED 引擎只吃 wav；Qwen 引擎也用它分段后的统一样式）"""
    import librosa, soundfile as sf
    y, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    p = tmp_dir / "input16k.wav"
    if p.parent != tmp_dir:
        raise RuntimeError("临时路径越出临时目录,已终止")
    sf.write(str(p), y, 16000)
    return str(p), y


def maybe_denoise(audio_path, enabled, log):
    """ZipEnhancer 前处理(可选);返回新路径字符串。注意:实测对白噪声未见收益,含 BGM/真实嘈杂再试"""
    if not enabled:
        return str(audio_path)
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    t0 = time.time()
    ans = pipeline(Tasks.acoustic_noise_suppression, model="iic/speech_zipenhancer_ans_multiloss_16k_base")
    out = Path(tempfile.mkdtemp(prefix="qwen_asr_denoise_")).resolve() / "enhanced.wav"
    ans(str(audio_path), output_path=str(out))
    log(f"[denoise] ZipEnhancer 前处理完成 ({time.time()-t0:.0f}s) -> {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Qwen3-ASR 本地转写（可说话人分离/字幕/专名纠错）")
    ap.add_argument("audio", help="音频文件路径(mp3/m4a/wav/flac/ogg/webm)")
    ap.add_argument("--diarize", action="store_true", help="启用说话人分离(fsmn-vad + cam++ + 聚类);长音频不加此参也会自动分段提速")
    ap.add_argument("--language", default=None,
                    help="强制语言:代码(zh/en/yue/ja...)或全名(Chinese/English...),默认自动检测")
    ap.add_argument("--hotwords", default=None, help="热词/上下文,逗号分隔,作为 system prompt 偏置识别")
    ap.add_argument("--replace", default=None, help="专名纠错,如 '开饭时间=>开放时间,小蜜=>小米'（确定性替换）")
    ap.add_argument("--replace-file", default=None, help="纠错词典文件(每行 '错=>对' 或 目标词)")
    ap.add_argument("--fuzzy", action="store_true", help="纠错启用拼音模糊匹配（需 pypinyin+rapidfuzz）")
    ap.add_argument("--srt", action="store_true", help="额外输出 .srt 字幕（加载 fa-zh 强制对齐出字级时间戳）")
    ap.add_argument("--max-line", type=int, default=28, help="单条字幕最大字数(默认28)")
    ap.add_argument("--names", default=None, help="说话人重命名,如 '0=张三,1=李四'")
    ap.add_argument("--merge-gap", type=int, default=800, help="相邻同说话人合并间隔阈值ms(默认800,设0关闭)")
    ap.add_argument("--vad", choices=["fsmn", "firered"], default="fsmn", help="VAD 后端(默认 fsmn;firered 需已装 fireredvad)")
    ap.add_argument("--engine", choices=["qwen", "aed"], default="qwen",
                    help="识别引擎: qwen=Qwen3-ASR-1.7B(默认,多语言含日文); aed=FireRedASR2-AED(中/英/粤更准,不支持日文,输出经 ct-punc 补标点)")
    ap.add_argument("--no-punc", action="store_true", help="aed 引擎不补标点(默认用 ct-punc 补)")
    ap.add_argument("--denoise", action="store_true", help="先跑 ZipEnhancer 降噪(含 BGM/强噪时试;白噪声实测无明显收益)")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="声纹余弦聚类阈值(声音相近漏分调低0.65-0.7,同人被拆调高0.8)")
    ap.add_argument("--batch-size", type=int, default=8, help="diarize 分段转写批大小")
    ap.add_argument("--max-new-tokens", type=int, default=2048, help="单段最大生成 token 数")
    ap.add_argument("--no-t2s", action="store_true", help="关闭默认的繁->简转换(保留模型原始输出)")
    ap.add_argument("--device", default="cuda:0", help="推理设备,无 N 卡用 cpu")
    args = ap.parse_args()

    # 输入路径校验:拒绝 ".."，realpath 规范化
    if ".." in args.audio:
        sys.exit("路径不允许包含 '..'")
    audio_path = Path(os.path.realpath(args.audio))
    if not audio_path.is_file():
        sys.exit(f"音频文件不存在: {audio_path}")
    # 输出路径锁定在音频同目录(派生后显式校验目录边界)
    out_json = audio_path.with_suffix(".json")
    out_srt = audio_path.with_suffix(".srt")
    for p in (out_json, out_srt):
        if p.parent != audio_path.parent:
            sys.exit("输出路径越出音频目录,已终止")
    hotwords = "、".join(w.strip() for w in args.hotwords.split(",") if w.strip()) if args.hotwords else None
    replace_map = {}
    if args.replace_file:
        p = Path(os.path.realpath(args.replace_file))
        if not p.is_file():
            sys.exit(f"纠错词典文件不存在: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                k, v = line.split("=>", 1)
                replace_map[k.strip()] = v.strip()
            else:
                replace_map[line] = line
    if args.replace:
        for item in args.replace.replace("；", ";").replace(";", ",").split(","):
            if "=>" in item:
                k, v = item.split("=>", 1)
                if k.strip() and v.strip():
                    replace_map[k.strip()] = v.strip()
    names = {}
    if args.names:
        for item in args.names.replace("；", ";").replace(";", ",").split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                if k.strip().isdigit() and v.strip():
                    names[int(k.strip())] = v.strip()

    t0 = time.time()
    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    use_aed = args.engine == "aed"
    work_dir = Path(tempfile.mkdtemp(prefix="qwen_asr_")).resolve()
    if args.denoise:
        src = maybe_denoise(audio_path, True, log)
    else:
        src = str(audio_path)
    import librosa, soundfile as sf
    preloaded_audio = None
    # AED 只吃 16k wav;为统一,两种引擎都先落到工作目录的 16k wav(顺便完成格式归一)
    if use_aed:
        src, preloaded_audio = to_wav16k(src, work_dir)
    if use_aed:
        model, processor, load_s = load_aed_model(args.device)
    else:
        model, processor, load_s = load_model(args.device)
    t2s = build_t2s(args.no_t2s)
    matcher = build_matcher(replace_map, args.fuzzy)
    punc = load_punc(args.device) if (use_aed and not args.no_punc) else None
    log(f"[1] {args.engine.upper()} 引擎已加载到 {args.device} ({load_s:.0f}s)"
        + ("" if t2s else "（繁->简已关闭）")
        + (f"；专名纠错 {len(replace_map)} 条" if replace_map else "")
        + ("；ct-punc 已加载" if punc is not None else ""))
    fa = None
    if args.srt:
        from funasr import AutoModel
        fa = AutoModel(model="fa-zh", device=args.device, disable_update=True)
        log("[1b] fa-zh 强制对齐模型已加载")

    def infer(paths):
        """统一识别入口（按引擎分发,批量）"""
        out = []
        bs = args.batch_size if use_aed else max(args.batch_size, 8)  # AED 批小些更稳
        for i in range(0, len(paths), bs):
            out.extend(transcribe_batch(model, processor, paths[i:i + bs], args.language,
                                        hotwords, args.max_new_tokens, t2s=t2s, matcher=matcher,
                                        punc=punc, engine=args.engine))
        return out

    _engine_tag = "FireRedASR2-AED" if use_aed else "Qwen3-ASR-1.7B"

    if not args.diarize:
        # ---- 整段转写;长音频(>60s)自动 VAD 分段批量(提速+每段语言独立检测),不标说话人 ----
        duration_s = sf.info(src).duration
        units, aligns = [], None
        if duration_s > AUTO_SEGMENT_S:
            segments = run_vad(src, args.vad, args.device)
            if not segments:
                sys.exit("未检测到语音(VAD 结果为空)")
            log(f"[2] 音频 {duration_s:.0f}s > {AUTO_SEGMENT_S}s, 自动 VAD 分段 {len(segments)} 段批量转写")
            audio = preloaded_audio if preloaded_audio is not None else librosa.load(src, sr=16000, mono=True)[0]
            units = [{"start_ms": int(s), "end_ms": int(e), "text": "", "spk": None, "seg_idx": [i]}
                     for i, (s, e) in enumerate(segments)]
            seg_files = write_unit_wavs(audio, units, work_dir)
            parsed = infer(seg_files)
            for u, p in zip(units, parsed):
                u["text"] = p["transcription"]
                u["language"] = p.get("language")
            texts = [u["text"] for u in units]
            if use_aed and aed_ts_usable(parsed):
                aligns = aed_aligns_from(parsed)
                log(f"[2b] 使用 AED 原生字级时间戳（{len(parsed)} 段）")
            elif fa is not None:
                aligns = align_batch(fa, seg_files, texts)
                if use_aed:
                    log("[2b] AED 时间戳不可用,已改用 fa-zh 对齐")
            print("\n".join(texts))
            langs = [p.get("language") for p in parsed if p.get("language")]
            main_lang = max(set(langs), key=langs.count) if langs else None
            payload = {"language": main_lang, "text": "\n".join(texts), "engine": _engine_tag,
                       "segments": [{"start_ms": u["start_ms"], "end_ms": u["end_ms"], "language": u.get("language"),
                                     "text": u["text"]} for u in units]}
            if aligns is not None:
                payload["sentences"] = subtitle_units(units, aligns, args.max_line)
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if aligns is not None:
                write_srt(payload["sentences"], out_srt, use_speaker=False)
            log(f"[3] 完成 (总耗时 {time.time()-t0:.0f}s), 结构化结果: {out_json}"
                + (f", 字幕: {out_srt}" if aligns is not None else ""))
            return
        res = infer([src])[0]
        lang = res.get("language") or ("aed" if use_aed else "auto")
        print(f"[{lang}] {res['transcription']}")
        u = {"start_ms": 0, "end_ms": int(duration_s * 1000), "text": res["transcription"],
             "spk": None, "spk_name": None}
        payload = {"language": res.get("language"), "text": res["transcription"],
                   "engine": _engine_tag}
        if use_aed and aed_ts_usable([res]):
            aligns = aed_aligns_from([res])
            log("[2b] 使用 AED 原生字级时间戳")
        elif fa is not None:
            aligns = align_batch(fa, [src], [res["transcription"]])
        else:
            aligns = None
        if aligns:
            payload["sentences"] = subtitle_units([u], aligns, args.max_line)
            write_srt(payload["sentences"], out_srt, use_speaker=False)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[2] 完成 (总耗时 {time.time()-t0:.0f}s), 结构化结果: {out_json}"
            + (f", 字幕: {out_srt}" if payload.get("sentences") else ""))
        return

    # ---- 说话人分离模式:分环实现(与技能 diarize.py 同一逻辑,识别引擎换成 Qwen3-ASR) ----
    from sklearn.cluster import AgglomerativeClustering

    segments = run_vad(src, args.vad, args.device)
    if not segments:
        sys.exit("未检测到语音(VAD 结果为空)")
    log(f"[2] VAD({args.vad}) 切出 {len(segments)} 个语音段 (累计 {time.time()-t0:.0f}s)")
    audio = preloaded_audio if preloaded_audio is not None else librosa.load(src, sr=16000, mono=True)[0]

    from funasr import AutoModel
    spk = AutoModel(model="cam++", device=args.device, disable_update=True)

    def to_numpy(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    embeddings = []
    for (s_ms, e_ms) in segments:
        seg = audio[s_ms * 16: e_ms * 16]
        emb = to_numpy(spk.generate(input=seg, fs=16000)[0]["spk_embedding"]).flatten()
        embeddings.append(emb / np.linalg.norm(emb))
    embeddings = np.array(embeddings)

    sim = embeddings @ embeddings.T
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    if len(segments) >= 2:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1 - args.threshold,
            metric="precomputed", linkage="average",
        ).fit_predict((dist + dist.T) / 2)
        log(f"[3] 声纹聚类完成: {labels.max()+1} 个说话人 (阈值 {args.threshold})")

        # 簇间相似度边缘提示:帮助发现"同人被拆分"(实测合成音频 5 人聚成 6 簇即此情形)
        uniq = sorted(set(labels.tolist()))
        if len(uniq) >= 2:
            pair_sims = []
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = embeddings[labels == uniq[i]], embeddings[labels == uniq[j]]
                    pair_sims.append((float((a @ b.T).mean()), uniq[i], uniq[j]))
            pair_sims.sort(reverse=True)
            top_sim, spk_a, spk_b = pair_sims[0]
            if top_sim >= args.threshold - 0.08:
                log(f"提示: 说话人{spk_a}与说话人{spk_b}声纹相似度 {top_sim:.2f}, 接近阈值 {args.threshold}; "
                    f"若实为同一人可试 --threshold {max(0.5, args.threshold - 0.05):.2f}")
    else:
        labels = np.zeros(len(segments), dtype=int)
        log("[3] 语音段不足 2 段,跳过聚类(视为单一说话人)")

    # 合并相邻同说话人碎段(预设 800ms,纪要可读性)
    units = merge_by_speaker(segments, labels, args.merge_gap)
    if args.merge_gap > 0 and len(units) < len(segments):
        log(f"[4] 碎段合并: {len(segments)} 段 -> {len(units)} 个发言块 (间隔阈值 {args.merge_gap}ms)")
    else:
        log(f"[4] 输出 {len(units)} 个发言块")

    tmp_dir = Path(tempfile.mkdtemp(prefix="qwen_asr_seg_")).resolve()
    unit_files = write_unit_wavs(audio, units, tmp_dir)

    parsed = infer(unit_files)
    for u, p in zip(units, parsed):
        u["text"] = p["transcription"]
        u["language"] = p.get("language")
        u["spk_name"] = names.get(u["spk"])
    aligns = None
    if use_aed and all(p.get("timestamp") for p in parsed):
        if all(len(p["timestamp"]) == len(p["transcription"]) for p in parsed):
            aligns = aed_aligns_from(parsed)
            log("[4b] 使用 AED 原生字级时间戳")
        elif fa is not None:
            aligns = align_batch(fa, unit_files, [u["text"] for u in units])
    elif fa is not None:
        aligns = align_batch(fa, unit_files, [u["text"] for u in units])
    log(f"[5] 转写完成 (累计 {time.time()-t0:.0f}s),结果:")

    lines = []
    for u in units:
        s_ms, e_ms = u["start_ms"], u["end_ms"]
        mm1, ss1 = divmod(s_ms // 1000, 60)
        mm2, ss2 = divmod(e_ms // 1000, 60)
        label = u["spk_name"] or f"说话人{u['spk']}"
        text = u["text"]
        print(f"{label} [{mm1:02d}:{ss1:02d}-{mm2:02d}:{ss2:02d}] {text}")
        lines.append({"spk": u["spk"], "spk_name": u["spk_name"], "start_ms": s_ms, "end_ms": e_ms,
                      "text": text, "language": u.get("language")})

    payload = {"engine": "Qwen3-ASR-1.7B", "audio": str(audio_path),
               "speakers": names or None, "lines": lines}
    if aligns is not None:
        subs = subtitle_units(units, aligns, args.max_line)
        payload["sentences"] = subs
        write_srt(subs, out_srt, use_speaker=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[6] 结构化结果已保存: {out_json}" + (f", 字幕: {out_srt}" if aligns is not None else "")
        + f"  (总耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
