#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端回归脚本：一条命令跑完整链路 + 断言产物齐全。

运行: python scripts/test_pipeline_e2e.py
原理：自动合成 6 秒测试视频（纯色画面 + TTS 中文语音），跑 av_to_transcript_and_minutes.py
      完整链路，断言各类产物存在且非空，最后清理现场。
覆盖：拆轨（无声视频/完整音轨）、人声分离、噪声门、说话人分离转写、文稿排版成 docx。
不改动任何既有产物：全部操作在系统临时目录中进行。
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _find_pipeline() -> pathlib.Path:
    """自动定位链路主脚本（兼容本机工作名）"""
    for name in ("av_to_transcript_and_minutes.py", "separate_video_audio.py"):
        p = SCRIPT_DIR / name
        if p.is_file():
            return p
    sys.exit("找不到链路主脚本（av_to_transcript_and_minutes.py / separate_video_audio.py）")


PIPELINE = _find_pipeline()

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        fails.append(name)


def make_tts_video(work: pathlib.Path) -> pathlib.Path:
    """合成测试素材：Windows TTS 中文语音 + 纯色画面（无外部依赖、可重复跑）"""
    wav = work / "voice.wav"
    ps = work / "tts.ps1"
    ps.write_text(
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$s.SelectVoice('Microsoft Huihui Desktop')\n"
        f"$s.SetOutputToWaveFile('{wav}')\n"
        "$s.Speak('今天天气很好，我们一起出门散步吧。')\n"
        "$s.Dispose()\n", encoding="utf-8-sig")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(ps)], check=True, capture_output=True)
    video = work / "e2e_test.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=navy:s=320x240:d=8",
                    "-i", str(wav), "-c:v", "libx264", "-c:a", "aac", "-shortest",
                    str(video)], check=True, capture_output=True)
    return video


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    work = pathlib.Path(tempfile.mkdtemp(prefix="pipeline_e2e_"))
    out_dir = work / "out"
    try:
        print("[1] 合成测试视频（TTS + 纯色画面）")
        video = make_tts_video(work)
        check("测试视频已生成", video.exists() and video.stat().st_size > 0)

        print("[2] 跑完整链路（默认模式：转写文稿）")
        r = subprocess.run([sys.executable, str(PIPELINE), str(video),
                            "-o", str(out_dir)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        check("链路退出码 0", r.returncode == 0, f"exit={r.returncode}\n{r.stdout[-500:]}")
        check("汇总显示完成 1", "完成 1" in (r.stdout or ""), (r.stdout or "")[-200:])

        stem_dir = out_dir / "e2e_test"
        expect = {
            "无声视频.mp4": "拆轨-无声视频",
            "完整音轨.m4a": "拆轨-完整音轨",
            "人声.wav": "人声分离",
            "背景音.wav": "背景音分离",
            "人声_转写用.wav": "噪声门中间件",
            "转写文稿-e2e_test.md": "转写文稿 md",
            "转写文稿-e2e_test.docx": "转写文稿 docx",
        }
        print("[3] 断言产物齐全且非空")
        for name, label in expect.items():
            p = stem_dir / name
            check(f"{label}", p.exists() and p.stat().st_size > 0,
                  f"缺失或为空: {p}")

        print("[4] 断言转写内容正确 + docx 有效")
        md = stem_dir / "转写文稿-e2e_test.md"
        if md.exists():
            text = md.read_text(encoding="utf-8")
            check("转写文本非空", len(text.strip()) > 20, f"len={len(text)}")
            check("含 TTS 原文关键词", "天气" in text or "散步" in text, text[:80])
        docx = stem_dir / "转写文稿-e2e_test.docx"
        check("docx 非空且为 zip 格式",
              docx.exists() and docx.read_bytes()[:2] == b"PK")

        print("[5] 断言幂等（重跑应跳过）")
        r2 = subprocess.run([sys.executable, str(PIPELINE), str(video),
                             "-o", str(out_dir)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        check("重跑跳过已处理", "跳过 1" in (r2.stdout or ""), (r2.stdout or "")[-200:])

        print("[6] 断言会议模式产物命名")
        r3 = subprocess.run([sys.executable, str(PIPELINE), str(video),
                             "-o", str(out_dir), "--meeting"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
        meeting_md = stem_dir / "会议原文-e2e_test.md"
        check("会议模式退出码 0", r3.returncode == 0, f"exit={r3.returncode}")
        check("会议原文已产出", meeting_md.exists() and meeting_md.stat().st_size > 0)
        if meeting_md.exists():
            mt = meeting_md.read_text(encoding="utf-8")
            check("会议原文含时间戳标签", "[00:0" in mt, mt[:60])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    if fails:
        print(f"FAILED: {len(fails)} 项 -> {fails}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
