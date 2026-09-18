# FunASR 快速转写（单说话人 / 不区分说话人）
# 用法: python transcribe.py <音频文件> [--language 中文|auto|English|...] [--vad] [--json 输出路径] [--no-itn] [--device cuda:0|cpu]
# 依赖: funasr, 模型 Fun-ASR-Nano-2512(首次运行自动下载约 2GB)
# 说明: 日常主力推荐 qwen_asr.py(Qwen3-ASR-1.7B,中文更准且支持字幕/说话人分离),
#       本脚本作为 FunASR 原生通道备选与对照基线保留。
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="FunASR 本地快速转写")
    ap.add_argument("audio", help="音频文件路径(mp3/m4a/wav/flac/ogg/webm)")
    ap.add_argument("--language", default="auto", help="语言: auto/中文/English/日文/粤语...")
    ap.add_argument("--vad", action="store_true", help="长音频启用 VAD 切分(超过约3分钟建议开启)")
    ap.add_argument("--json", dest="json_out", default=None, help="额外保存结构化结果到此 JSON 路径")
    ap.add_argument("--no-itn", action="store_true", help="关闭文本规整(数字/日期归一)")
    ap.add_argument("--device", default="cuda:0", help="推理设备,无 N 卡用 cpu")
    args = ap.parse_args()

    # 输入路径校验:拒绝 ".."，realpath 规范化(与 qwen_asr.py 同一安全模式)
    if ".." in args.audio:
        sys.exit("路径不允许包含 '..'")
    audio_path = Path(os.path.realpath(args.audio))
    if not audio_path.is_file():
        sys.exit(f"音频文件不存在: {audio_path}")

    t0 = time.time()
    from funasr import AutoModel

    kw = dict(model="FunAudioLLM/Fun-ASR-Nano-2512", trust_remote_code=True, device=args.device)
    if args.vad:
        kw.update(vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000})
    model = AutoModel(**kw)

    res = model.generate(
        input=[str(audio_path)], cache={}, batch_size=1,
        language=args.language, itn=not args.no_itn,
    )
    text = res[0].get("text", "").strip()
    dur = time.time() - t0
    print(text)
    print(f"[转写完成 {dur:.1f}s]", file=sys.stderr)

    if args.json_out:
        # 输出路径同样校验:拒绝 ".." + realpath 规范化后落盘
        if ".." in args.json_out:
            sys.exit("输出路径不允许包含 '..'")
        json_path = Path(os.path.realpath(args.json_out))
        json_path.write_text(json.dumps({"audio": str(audio_path), "text": text,
                                         "language": args.language, "elapsed_s": round(dur, 1)},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[JSON 已保存: {json_path}]", file=sys.stderr)

if __name__ == "__main__":
    main()
