# 带说话人分离的转写（手动分环实现,规避 funasr 1.4.8 spk_model pipeline 聚类缺陷）
# 流程: fsmn-vad 切段 -> 每段 cam++ 声纹 -> 余弦阈值层次聚类 -> Fun-ASR-Nano 分段转写
# 用法: python diarize.py <音频文件> [--threshold 0.75] [--language auto] [--device cuda:0]
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="FunASR 说话人分离转写")
    ap.add_argument("audio", help="音频文件路径(mp3/m4a/wav/flac/ogg/webm)")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="声纹余弦相似度阈值:低于此值判为不同人(声音相近漏分时调低0.65-0.7,同人被拆分时调高0.8)")
    ap.add_argument("--language", default="auto", help="转写语言")
    ap.add_argument("--device", default="cuda:0", help="推理设备,无 N 卡用 cpu")
    args = ap.parse_args()

    # 输入路径校验:拒绝 ".."，realpath 规范化(与 qwen_asr.py 同一安全模式)
    if ".." in args.audio:
        sys.exit("路径不允许包含 '..'")
    audio_path = Path(os.path.realpath(args.audio))
    if not audio_path.is_file():
        sys.exit(f"音频文件不存在: {audio_path}")

    t0 = time.time()
    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    import librosa, soundfile as sf
    from funasr import AutoModel

    vad = AutoModel(model="fsmn-vad", device=args.device)
    spk = AutoModel(model="cam++", device=args.device)
    asr = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", trust_remote_code=True, device=args.device)
    log(f"[1] 三个模型已加载到 {args.device} (累计 {time.time()-t0:.0f}s)")

    # ---- 1. VAD 切段 ----
    segments = vad.generate(input=str(audio_path))[0]["value"]  # [[start_ms, end_ms], ...]
    log(f"[2] VAD 切出 {len(segments)} 个语音段")
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # ---- 2. 每段声纹 embedding ----
    def to_numpy(x):  # GPU tensor 需先回内存
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    embeddings = []
    for (s_ms, e_ms) in segments:
        seg = audio[s_ms * 16 : e_ms * 16]
        emb = to_numpy(spk.generate(input=seg, fs=16000)[0]["spk_embedding"]).flatten()
        embeddings.append(emb / np.linalg.norm(emb))
    embeddings = np.array(embeddings)

    # ---- 3. 层次聚类(余弦距离阈值) ----
    sim = embeddings @ embeddings.T
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    from sklearn.cluster import AgglomerativeClustering
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - args.threshold,
        metric="precomputed", linkage="average",
    ).fit_predict((dist + dist.T) / 2)
    log(f"[3] 声纹聚类完成: {labels.max()+1} 个说话人 (阈值 {args.threshold})")

    # ---- 4. 分段转写(逐段写临时 wav 后批量传文件路径) ----
    tmp_dir = tempfile.mkdtemp(prefix="funasr_seg_")
    seg_files = []
    for i, (s_ms, e_ms) in enumerate(segments):
        p = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
        sf.write(p, audio[s_ms * 16 : e_ms * 16], 16000)
        seg_files.append(p)
    asr_res = asr.generate(input=seg_files, cache={}, batch_size=1,
                           language=args.language, itn=True)
    log(f"[4] 转写完成 (累计 {time.time()-t0:.0f}s),结果:")

    lines = []
    for i, (s_ms, e_ms) in enumerate(segments):
        text = asr_res[i].get("text", "")
        mm1, ss1 = divmod(s_ms // 1000, 60)
        mm2, ss2 = divmod(e_ms // 1000, 60)
        line = f"说话人{labels[i]} [{mm1:02d}:{ss1:02d}-{mm2:02d}:{ss2:02d}] {text}"
        print(line)  # stdout 交付
        lines.append({"spk": int(labels[i]), "start_ms": int(s_ms), "end_ms": int(e_ms), "text": text})

    # ---- 5. 结构化结果落盘(音频同目录;派生路径先做 ".." 与目录边界双重校验再写) ----
    out_json = audio_path.with_name(audio_path.stem + "_transcript.json")
    if ".." in str(out_json) or out_json.parent != audio_path.parent:
        sys.exit("输出路径越出音频目录,已终止")
    out_json.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[5] 结构化结果已保存: {out_json}  (总耗时 {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
