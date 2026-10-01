#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音视频 → 转写文稿 / 会议纪要 全自动批量工具（拆轨 + Demucs 人声分离 + 说话人分离转写 + 规范 docx 排版）

流程（视频与音频通用；转写恒走说话人分离）：
  视频 → 无声视频 + 完整音轨（均无损 copy）；音频输入跳过本环（源即完整音频）
  音频/完整音轨 → 人声.wav + 背景音.wav（Demucs 分离）
  人声 → 文字（qwen_asr 恒走 --diarize：VAD 切段 + 声纹聚类说话人）
  文字 → 转写文稿（默认，每4句一段）或 会议原文（--meeting，按时间戳完整转写）
         → article-format 规范 docx；会议纪要由会话内 AI 从会议原文提炼

每个输入产出（同一子目录）：
  无声视频.<原扩展名>           视频流无损提取（仅视频输入）
  完整音轨.<按源音轨编码>       原音轨无损提取（仅视频输入）
  人声.wav / 背景音.wav         从音频中分离出的人声 / 背景音（分离产物）
  人声_转写用.wav               清除静音伪影后供转写的人声（可复现的中间件）
  转写结果.json                 转写结构化结果（说话人分段 / 文本 / 时间戳）
  转写文稿-<名>.md / .docx      转写文稿（默认模式）
  会议原文-<名>.md / .docx      会议原文（--meeting；正式纪要由 AI 从原文提炼为「会议纪要-<名>」）

用法：
  python av_to_transcript_and_minutes.py <视频/音频文件或文件夹> [-o 输出目录] [--force] [--no-asr] [--meeting]

常用参数：
  --meeting               产出带说话人与时间戳的会议原文（正式纪要由会话内 AI 提炼）
  --no-separate           纯人声素材跳过 Demucs 分离，直接转写（更快）
  --names "0=张三,1=李四"  固定说话人姓名（先听一段确认谁是谁）
  --srt                   额外产出字级时间戳字幕
  --asr-engine aed        换识别引擎（中/英/粤更准）；换引擎会自动用新缓存重转，不复用旧结果
  --replace "错=>对"       确定性专名纠错（同音异字可配 --asr-extra "--fuzzy"），比热词偏置可靠
  --denoise               转写前 ZipEnhancer 降噪（仅真实含 BGM/强噪素材）
  --vad firered           改 VAD 后端（默认 fsmn）
  --asr-extra "--min-seg 300"  其余 qwen_asr 参数原样透传
  --verbose               实时透传子进程输出，排查卡顿与失败原因用

完成判定与续跑：
  md 与 docx 齐备才算完成；只有 md 而缺 docx（首跑时未装 article-format）时只补排版，
  不重跑分离与转写。转写按配置分文件缓存，阈值/热词/人名/字幕/识别引擎/额外参数任一不同
  即互不复用；--force 才全部重跑。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SENTS_PER_PARA = 4  # 转写文稿分段：每几句拼一段
# 管线自身保留的 qwen_asr 参数，禁止经 --asr-extra 二次传入（否则重复传参会破坏调用契约）
RESERVED_ASR_ARGS = {"--job-config", "--diarize", "--audio"}
# 经任务配置 JSON 透传给 qwen_asr 的键（其余参数一律走命令行 extra，避免"未知项"报错）
ASR_JOB_KEYS = ("threshold", "diarize_engine", "hotwords", "names", "srt", "engine")

_asr_script_cache = None


def child_env(**overrides) -> dict:
    """子进程环境：把 Python 侧标准输出锁死为 UTF-8。
    中文 Windows 默认 cp936，父进程若按 cp936 解码子进程的 UTF-8 字节会抛
    UnicodeDecodeError（reader 线程崩掉、捕获输出静默变 None），故子进程输出编码
    与父进程解码编码两端必须同时锁定，缺一端就丢日志。"""
    env = os.environ | {"PYTHONIOENCODING": "utf-8",
                        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    env.update(overrides)
    return env


def run_child(cmd: list, verbose: bool = False, env: dict = None):
    """统一子进程调用：默认捕获输出并按 UTF-8 解码（errors=replace 兜底不崩）；
    verbose=True 时输出实时透传终端（排查卡顿与失败原因用）"""
    if verbose:
        return subprocess.run(cmd, check=True, env=env, shell=False)
    return subprocess.run(cmd, check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, shell=False)


def run_output(cmd: list, env: dict = None) -> str:
    """取一次性短输出（如 ffprobe 查询），固定 UTF-8 解码"""
    r = subprocess.run(cmd, check=False, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, shell=False)
    return r.stdout


def split_asr_extra(text: str) -> list:
    """分割 --asr-extra 参数串：按空白切分，支持单/双引号包裹含空格的值。
    刻意不用 shlex：其 posix 模式把反斜杠当转义符，Windows 路径
    （如 --replace-file D:\\dict.txt）会被吃掉一个字符。"""
    tokens, buf, quote = [], [], None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
        elif ch == "'" or ch == '"':
            quote = ch
        elif ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def asr_script() -> Path:
    """定位转写引擎 qwen_asr.py（惰性查找：仅在真正转写时调用，不影响 --help）：
    环境变量 VOICE_ASR_SCRIPT → 同目录 → 技能目录（本仓库布局）"""
    global _asr_script_cache
    if _asr_script_cache is not None:
        return _asr_script_cache
    env = os.environ.get("VOICE_ASR_SCRIPT")
    cands = ([Path(env)] if env else []) + [
        SCRIPT_DIR / "qwen_asr.py",
        SCRIPT_DIR / "skills" / "funasr-transcribe" / "scripts" / "qwen_asr.py"]
    for c in cands:
        if c.is_file():
            _asr_script_cache = c
            return c
    sys.exit("找不到 qwen_asr.py：请用环境变量 VOICE_ASR_SCRIPT 指定其路径")


_article_skill_cache = None


def article_skill() -> Path:
    """定位「全局文章排版规范」技能脚本目录（article-format，惰性查找）：
    环境变量 ARTICLE_SKILL_DIR → 常见技能安装位置；找不到时返回 None（跳过排版，产出标准 docx）"""
    global _article_skill_cache
    if _article_skill_cache is not None:
        return _article_skill_cache or None
    env = os.environ.get("ARTICLE_SKILL_DIR")
    cands = ([Path(env)] if env else []) + [
        Path.home() / ".zcode" / "skills" / "article-format" / "scripts",
        Path.home() / ".dsh" / "skills" / "article-format" / "scripts",
        Path.home() / ".agents" / "skills" / "article-format" / "scripts",
        Path.home() / ".claude" / "skills" / "article-format" / "scripts",
        SCRIPT_DIR / "skills" / "article-format" / "scripts"]
    for c in cands:
        if (c / "md2docx.py").is_file():
            _article_skill_cache = c
            return c
    _article_skill_cache = False
    return None

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts",
              ".wmv", ".m4v", ".3gp", ".mpg", ".mpeg"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr"}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS
# 源音轨编码 → 无损 copy 时的容器扩展名（codec 名不带点；aac 之外的冷门编码落 mka 万能容器）
AUDIO_EXT = {"aac": ".m4a", "mp3": ".mp3", "flac": ".flac",
             "opus": ".opus", "vorbis": ".ogg", "ac3": ".ac3", "eac3": ".eac3"}


def probe_audio_codec(video: Path) -> str:
    return run_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)]).strip().lower()


def asr_cache_name(threshold=None, diarize_engine=None, hotwords=None, names=None,
                   srt=False, engine=None, extra=None) -> str:
    """转写缓存文件名：任一影响结果的配置不同 → 缓存不同。
    识别引擎（qwen/aed）输出的文本不同，是缓存键的必需维度——漏掉它会让人以为换了引擎、
    实际拿到的是旧引擎结果；热词、人名、字幕、额外 ASR 参数同理不可与默认结果混用。"""
    parts = []
    if engine == "aed":
        parts.append("aed")
    if diarize_engine == "pyannote":
        parts.append("pyannote")
    if threshold is not None:
        parts.append(f"阈值{threshold}")
    if hotwords:
        parts.append("热词" + hashlib.sha256(hotwords.encode("utf-8")).hexdigest()[:6])
    if names:
        parts.append("人名" + hashlib.sha256(names.encode("utf-8")).hexdigest()[:6])
    if srt:
        parts.append("字幕")
    if extra:
        parts.append("扩展" + hashlib.sha256(" ".join(extra).encode("utf-8")).hexdigest()[:6])
    return "转写结果" + ("_" + "_".join(parts) if parts else "") + ".json"


def call_asr(vocals: Path, opts: dict, force: bool = False):
    """调 qwen_asr.py 转写（恒走 --diarize：说话人分离），返回结构化 JSON。
    opts 键：threshold / diarize_engine / hotwords / names / srt / engine（识别引擎）
             extra（额外 ASR 参数列表，命令行追加）/ verbose（子进程输出实时透传）
    缓存已存在且非 --force 时直接复用（转写是最贵环节，支持断点续跑）；
    白名单键走任务配置 JSON，其余额外参数以命令行形式追加。"""
    extra = list(opts.get("extra") or [])
    cache_json = vocals.parent / asr_cache_name(
        opts.get("threshold"), opts.get("diarize_engine"), opts.get("hotwords"),
        opts.get("names"), bool(opts.get("srt")), opts.get("engine"), extra)
    if cache_json.exists() and not force:
        return json.loads(cache_json.read_text(encoding="utf-8"))
    raw_json = vocals.parent / (vocals.stem + ".json")  # qwen_asr 固定输出名：<音频名>.json
    raw_srt = vocals.with_suffix(".srt")
    cfg_path = vocals.parent / "_asr_job.json"
    # 只透传 qwen_asr 认识的键：opts 里的 extra / verbose 不是它的参数，混进去会被判"未知项"退出
    cfg = {k: opts[k] for k in ASR_JOB_KEYS if opts.get(k) not in (None, False)}
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    raw_json.unlink(missing_ok=True)
    raw_srt.unlink(missing_ok=True)
    run_child([sys.executable, str(asr_script()), str(vocals), "--diarize",
               "--job-config", str(cfg_path)] + extra,
              verbose=bool(opts.get("verbose")), env=child_env())
    cfg_path.unlink(missing_ok=True)
    raw_json.replace(cache_json)
    return json.loads(cache_json.read_text(encoding="utf-8"))


def transcript_md(payload: dict, stem: str) -> str:
    """默认模式：转写文稿——从说话人分段结构拼全文，按 4 句一段成稿（不带说话人标签）"""
    lines = payload.get("lines") or []
    text = "\n".join((l.get("text") or "").strip() for l in lines).strip()
    if not text:
        text = (payload.get("text") or "").strip()  # 兜底非 diarize 结构
    if not text:
        return ""
    sents = [s.strip() for s in re.split(r"(?<=[。！？!?])", text.replace("\n", "")) if s.strip()]
    paras = ["".join(sents[i:i + SENTS_PER_PARA]) for i in range(0, len(sents), SENTS_PER_PARA)]
    return f"# 转写文稿：{stem}\n\n" + "\n\n".join(paras) + "\n"


def meeting_md(payload: dict, stem: str) -> str:
    """会议模式：会议原文——按时间顺序完整转写，每段带 [分:秒] 时间戳与说话人标签
    （pyannote 分离可靠后恢复标签；--names 提供人名映射时显示真实姓名）
    供对照审核与纪要提炼；正式纪要由会话内 AI 按 meeting-notes-expert 模板从原文提炼"""
    body = []
    for l in payload.get("lines") or []:
        name = l.get("spk_name") or f"说话人{l.get('spk')}"
        m, s = divmod(int(l.get("start_ms", 0)) // 1000, 60)
        text = (l.get("text") or "").strip()
        if text:
            body.append(f"[{m:02d}:{s:02d}] {name}：{text}")
    joined = "\n\n".join(body) if body else "（未检测到语音）"
    return f"# 会议原文：{stem}\n\n{joined}\n"


def render_docx(md_path: Path, docx_path: Path, stem_dir: Path, verbose: bool = False) -> bool:
    """md -> article-format 规范 docx；技能缺失时只留 md 并提示，返回 False"""
    skill = article_skill()
    if skill is None:
        # 未安装 article-format 技能：跳过规范排版，仅保留 md（并在日志说明）
        print("[提示] 未找到 article-format 技能，跳过 docx 排版（仅产出 md）；"
              "如需规范排版请设置 ARTICLE_SKILL_DIR", file=sys.stderr)
        return False
    base = stem_dir / "_article_base_tmp.docx"
    # 主标题由 md 内首个 `# ` 行自动识别，无需 --title；
    # article-format 输出受 ARTICLE_OUT_DIR 约束，且两脚本对相对路径解析基准不同——传入路径一律绝对化
    art_env = child_env(ARTICLE_OUT_DIR=str(stem_dir.resolve()))
    run_child([sys.executable, str(skill / "md2docx.py"),
               str(md_path.resolve()), str(base.resolve())], verbose=verbose, env=art_env)
    run_child([sys.executable, str(skill / "article_style.py"),
               str(base.resolve()), str(docx_path.resolve())], verbose=verbose, env=art_env)
    base.unlink(missing_ok=True)
    return True


def transcribe_and_write_article(stem_dir: Path, stem: str, force: bool,
                                 meeting: bool = False, opts: dict = None) -> str:
    """人声 wav -> qwen_asr 转写 -> md（转写文稿/会议原文）-> article-format 排版 docx
    opts 键：threshold / diarize_engine / hotwords / names / srt / engine / extra / verbose
    返回 'done' / 'skip' / 'novoice' / 'redocx'（md 已在，仅补排版）"""
    opts = opts or {}
    out_stem = f"会议原文-{stem}" if meeting else f"转写文稿-{stem}"
    md_path = stem_dir / f"{out_stem}.md"
    docx_path = stem_dir / f"{out_stem}.docx"
    if md_path.exists() and not force:
        # md 已在：docx 齐备或排版技能缺失才算完成；只差 docx（首跑时技能未装）则只补排版，
        # 不重跑分离与转写——旧行为要到 --force 全量重跑，连 Demucs 一起白烧
        if docx_path.exists() or article_skill() is None:
            return "skip"
        print(f"[补排版] {stem}：md 已在，仅补 article-format docx")
        render_docx(md_path, docx_path, stem_dir, bool(opts.get("verbose")))
        return "redocx"
    vocals = stem_dir / "人声.wav"
    # demucs 在静音段输出低电平伪影（非真静音），VAD 能量门限会失效切不出段——
    # 先过噪声门：低于 -40dB 归零、人声直通，恢复段间真静音（实测 VAD 1 段 → 2 段）
    vocals_asr = stem_dir / "人声_转写用.wav"
    if force or not vocals_asr.exists():
        print(f"[噪声门] {stem}：agate 清理分离伪影")
        run_child(["ffmpeg", "-y", "-i", str(vocals), "-af",
                   "agate=threshold=-40dB:ratio=99:attack=2:release=100",
                   str(vocals_asr)], env=child_env())
    print(f"[转写] {stem}：说话人分离转写中（首次较慢；命中缓存直接复用）")
    payload = call_asr(vocals_asr, opts, force)
    md_body = meeting_md(payload, stem) if meeting else transcript_md(payload, stem)
    if not md_body:
        return "novoice"
    md_path.write_text(md_body, encoding="utf-8")
    if opts.get("srt"):  # qwen_asr 产出的字幕转移到本次输出名
        raw_srt = vocals_asr.with_suffix(".srt")
        if raw_srt.exists():
            raw_srt.replace(stem_dir / f"{out_stem}.srt")
    if render_docx(md_path, docx_path, stem_dir, bool(opts.get("verbose"))):
        print(f"[排版] {stem}：article-format 规范 docx 已生成")
    return "done"


def separate_one(media: Path, out_dir: Path, force: bool, do_asr: bool = True,
                 meeting: bool = False, opts: dict = None,
                 no_separate: bool = False) -> str:
    opts = opts or {}
    stem_dir = out_dir / media.stem
    is_audio = media.suffix.lower() in AUDIO_EXTS
    separated = (stem_dir / "人声.wav").exists() and (stem_dir / "背景音.wav").exists()
    out_stem = f"会议原文-{media.stem}" if meeting else f"转写文稿-{media.stem}"
    md_exists = (stem_dir / f"{out_stem}.md").exists()
    docx_exists = (stem_dir / f"{out_stem}.docx").exists()
    # md 为完成标记；但只有 md 而缺 docx（首跑时未装 article-format）不算完成，
    # 交给下游走"仅补排版"分支，避免补装技能后只能 --force 全量重跑
    article = md_exists and (docx_exists or article_skill() is None)
    if do_asr:
        all_done = article
    else:
        # --no-separate 只产出「人声.wav」（无背景音），完成判定要对齐实际产物
        all_done = (stem_dir / "人声.wav").exists() if no_separate else separated
    if all_done and not force:
        return "skip"
    stem_dir.mkdir(parents=True, exist_ok=True)

    # 1) FFmpeg 拆轨（仅视频输入；音频输入源即完整音频，跳过）
    if not is_audio:
        video_out = stem_dir / f"无声视频{media.suffix}"
        audio_ext = AUDIO_EXT.get(probe_audio_codec(media), ".mka")
        audio_out = stem_dir / f"完整音轨{audio_ext}"
        if force or not video_out.exists():
            print(f"[拆轨] {media.name}：提取视频流（无损 copy）")
            run_child(["ffmpeg", "-y", "-i", str(media), "-an", "-c:v", "copy",
                       str(video_out)], env=child_env())
        if force or not audio_out.exists():
            print(f"[拆轨] {media.name}：提取完整音轨（无损 copy）")
            run_child(["ffmpeg", "-y", "-i", str(media), "-vn", "-c:a", "copy",
                       str(audio_out)], env=child_env())

    # 2) Demucs 人声/背景音分离（--no-separate 时跳过：纯人声音频直接转码为人声.wav 供下游复用）
    if no_separate:
        vocals_out = stem_dir / "人声.wav"
        if force or not vocals_out.exists():
            print(f"[转码] {media.name}：跳过人声分离，直接转 44.1kHz 立体声 wav")
            run_child(["ffmpeg", "-y", "-i", str(media), "-ac", "2", "-ar", "44100",
                       str(vocals_out)], env=child_env())
    elif force or not separated:
        tmp = out_dir / "_demucs_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[分离] {media.name}：Demucs 人声/背景音分离（GPU，耗时随时长增长）")
        # demucs 的进度条有实际参考价值，始终实时透传（不捕获）
        run_child(["demucs", "-n", "htdemucs", "--two-stems=vocals",
                   "-o", str(tmp), str(media)], verbose=True, env=child_env())
        # demucs 固定输出 vocals.wav / no_vocals.wav，移入输出目录时改成自说明中文名
        rename = {"vocals.wav": "人声.wav", "no_vocals.wav": "背景音.wav"}
        for f in (tmp / "htdemucs" / media.stem).glob("*.wav"):
            shutil.move(str(f), str(stem_dir / rename.get(f.name, f.name)))
        shutil.rmtree(tmp, ignore_errors=True)

    # 3) 人声转文字 + 文章/纪要排版（--no-asr 可跳过）
    if do_asr:
        return transcribe_and_write_article(stem_dir, media.stem, force, meeting, opts)
    return "done"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="音视频全自动批量（拆轨 + 人声分离 + 说话人分离转写 → 转写文稿 / 会议纪要）")
    ap.add_argument("input", help="视频/音频文件或包含它们的文件夹")
    ap.add_argument("-o", "--output", help="输出目录（默认 <输入>/separated_out）")
    ap.add_argument("--force", action="store_true", help="已处理过的也重跑")
    ap.add_argument("--no-asr", action="store_true", help="只拆轨+人声分离，不转写不生成文稿")
    ap.add_argument("--no-separate", action="store_true",
                    help="跳过人声/背景音分离（纯人声音频适用，直接进转写）")
    ap.add_argument("--meeting", action="store_true",
                    help="产出会议原文（带时间戳；默认产出转写文稿。转写恒带说话人分离）")
    ap.add_argument("--diarize-engine", choices=["campp", "pyannote", "auto"], default="auto",
                    help="说话人分离引擎：auto=会议自动用 pyannote（重叠语音更准），其余用 campp；"
                         "campp=轻量声纹聚类；pyannote=分割模型（需已装）")
    ap.add_argument("--threshold", type=float, default=None,
                    help="声纹聚类阈值（仅 campp 引擎，默认 0.75）：单人素材被误切成多人时调低试 0.6~0.65；"
                         "配合 --force 生效，不同配置存独立缓存")
    ap.add_argument("--hotwords", default=None,
                    help="热词/上下文（逗号分隔），偏置识别引擎提升专名准确率，如 "
                         "'胚布,白配检测,库龄机制'；变更热词后自动用新缓存重转")
    ap.add_argument("--names", default=None,
                    help="说话人改名（如 '0=赵总,1=张会计'）——会议原文的说话人标签将显示真实姓名")
    ap.add_argument("--srt", action="store_true",
                    help="额外产出 .srt 字幕（同名，含字级时间戳；加载 fa-zh 对齐模型）")
    ap.add_argument("--asr-engine", choices=["qwen", "aed"], default="qwen",
                    help="识别引擎：qwen=Qwen3-ASR-1.7B（默认，多语言）；aed=FireRedASR2-AED（中/英/粤更准）")
    ap.add_argument("--replace", default=None,
                    help="专名/错词确定性纠错（如 '小蜜=>小米,开饭时间=>开放时间'）——"
                         "比 --hotwords 的 prompt 偏置可靠；变更后自动用新缓存重转")
    ap.add_argument("--denoise", action="store_true",
                    help="转写前跑 ZipEnhancer 降噪（仅真实含 BGM/强噪素材；干净素材不加更好）")
    ap.add_argument("--vad", choices=["fsmn", "firered"], default=None,
                    help="VAD 切段后端（默认 fsmn；firered 误报更低，需已装 fireredvad）")
    ap.add_argument("--language", default=None,
                    help="强制语言（zh/en/yue/ja 等代码），默认自动检测")
    ap.add_argument("--max-line", type=int, default=None,
                    help="单条字幕最大字数（默认 28，配合 --srt 使用）")
    ap.add_argument("--merge-gap", type=int, default=None,
                    help="相邻同说话人合并间隔 ms（默认 800，0 关闭）")
    ap.add_argument("--asr-extra", default=None,
                    help="其余 qwen_asr 参数原样透传（如 '--fuzzy --min-seg 300'）；"
                         "变更后自动用新缓存重转")
    ap.add_argument("--verbose", action="store_true",
                    help="实时透传各子进程输出（排查卡顿与失败原因时用）")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"输入不存在：{src}")
    out_dir = Path(args.output) if args.output else (
        src.parent / "separated_out" if src.is_file() else src / "separated_out")
    if src.is_file():
        if src.suffix.lower() not in MEDIA_EXTS:
            sys.exit(f"不支持的文件类型：{src.suffix}")
        medias = [src]
    else:
        medias = sorted(p for p in src.rglob("*") if p.suffix.lower() in MEDIA_EXTS)
    if not medias:
        sys.exit("没找到视频/音频文件")

    # auto：会议模式走 pyannote（重叠语音更准），其余走 campp（轻量）
    diarize_engine = ("pyannote" if args.meeting else None) \
        if args.diarize_engine == "auto" else args.diarize_engine
    # 额外 ASR 参数：显式项 + --asr-extra 合并，统一进命令行与缓存键
    extra = []
    for flag, val in (("--replace", args.replace), ("--language", args.language),
                      ("--max-line", args.max_line), ("--merge-gap", args.merge_gap),
                      ("--vad", args.vad)):
        if val is not None:
            extra += [flag, str(val)]
    if args.denoise:
        extra.append("--denoise")
    if args.asr_extra:
        extra += split_asr_extra(args.asr_extra)
    for bad in RESERVED_ASR_ARGS:
        if bad in extra:
            sys.exit(f"--asr-extra 不允许传入管线保留参数：{bad}")
    opts = {"threshold": args.threshold, "diarize_engine": diarize_engine,
            "hotwords": args.hotwords, "names": args.names, "srt": args.srt,
            "engine": args.asr_engine if args.asr_engine != "qwen" else None,
            "extra": extra, "verbose": args.verbose}

    ok = skip = fail = novoice = redocx = 0
    for v in medias:
        try:
            r = separate_one(v, out_dir, args.force, do_asr=not args.no_asr,
                             meeting=args.meeting, opts=opts,
                             no_separate=args.no_separate)
            if r == "skip":
                skip += 1
                print(f"[跳过] {v.name}（已处理过，--force 可重跑）")
            elif r == "novoice":
                novoice += 1
                print(f"[无人声] {v.name}（转写为空，仅出拆轨+分离件）")
            elif r == "redocx":
                redocx += 1
                print(f"[补排版] {v.name} → {out_dir / v.stem}")
            else:
                ok += 1
                print(f"[完成] {v.name} → {out_dir / v.stem}")
        except subprocess.CalledProcessError as e:
            fail += 1
            # 打印失败尾部 20 行：原先只留最后一行，真正原因往往在上面几行；
            # 输出已按 UTF-8 解码，不会因 GBK 错位丢成空
            detail = (e.stderr or e.stdout or "").strip()
            if detail:
                tail = "\n".join(detail.splitlines()[-20:])
                print(f"[失败] {v.name}：\n{tail}")
            else:
                print(f"[失败] {v.name}：{e}")
        except KeyboardInterrupt:
            print("\n用户中断"); break
        except Exception as e:
            # 兜底：缺 ffmpeg/demucs（FileNotFoundError）、缓存 JSON 损坏等不再中断整批
            fail += 1
            print(f"[失败] {v.name}：{type(e).__name__}: {e}")

    print(f"\n汇总：完成 {ok} / 补排版 {redocx} / 无人声 {novoice} / 跳过 {skip} / 失败 {fail}，输出目录：{out_dir}")


if __name__ == "__main__":
    main()
