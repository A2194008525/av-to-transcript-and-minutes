---
name: funasr-transcribe
description: "本地语音转文字（离线、GPU 加速、无需 API 密钥）：录音/音频转文字、会议访谈转写、说话人分离（谁在何时说了什么）、字幕 SRT 生成（字级时间戳）、专名纠错、无标点文本恢复标点、录音情绪分析。主力 Qwen3-ASR-1.7B + FunASR VAD/声纹，备选 Fun-ASR-Nano。Use whenever the user mentions transcribing a recording, speech-to-text, audio-to-text, meeting or interview transcription, subtitles/SRT from audio, speaker diarization, punctuation restoration, proper-noun correction, or emotion analysis of audio — even if they just drop an audio file and say 转一下 or 整理成文字."
---

# FunASR / Qwen3-ASR 本地语音转写

把录音转成文字的本地工具箱，GPU 加速、完全离线、无需 API 密钥。

**环境准备**（一次性）：Python 3.10+，`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130`，再 `pip install funasr modelscope`。完整安装步骤、版本要求（RTX 50 系显卡必须 cu128+）与故障修复见 `references/advanced.md` §五。模型首次运行时自动下载（Qwen3-ASR-1.7B 约 4.1GB），缓存在 `~/.cache/modelscope/models/`。

## 快速决策（按需求选脚本）

| 用户需求 | 用法 |
|---------|------|
| **日常转写（首选）**：单人或不需要区分说话人 | `python <技能目录>/scripts/qwen_asr.py <音频文件>` |
| 会议/访谈，要区分"谁说了什么" | `python .../qwen_asr.py <音频文件> --diarize` |
| 要字幕文件（SRT，可带说话人标签） | 加 `--srt`（fa-zh 强制对齐，字级时间戳） |
| 专有名词/人名总听错 | 加 `--replace "错词=>对词"`（确定性替换） |
| 录音嘈杂/带背景音乐 | 加 `--denoise`（ZipEnhancer 前处理，见约束 5） |
| 只要 FunASR 原生通道（对照/低显存兜底） | `python <技能目录>/scripts/transcribe.py <音频文件>` |
| 给无标点的转写文本补标点 | 见 references/advanced.md 的 ct-punc 节 |
| 分析录音/某段音频的情绪 | 见 references/advanced.md 的 emotion2vec 节 |

`<技能目录>` = 本 SKILL.md 所在目录。脚本直接用 `python` 跑。

## 主力脚本 qwen_asr.py（双引擎，Qwen3-ASR 默认 / FireRedASR2-AED 可选）

识别引擎二选一，VAD/声纹/字幕/纠错各引擎通用：

| 引擎 | 参数 | 强项 | 弱项 |
|------|------|------|------|
| Qwen3-ASR-1.7B（默认） | `--engine qwen` | 多语言（含**日文**）、自带标点、自带语种识别 | 中文偶有同音误听 |
| FireRedASR2-AED | `--engine aed` | 中/英/粤准确率更高、原生字级时间戳+置信度 | **不支持日文**、输出无标点（脚本自动用 ct-punc 补）、只吃 wav（脚本自动转） |

实测对照（2026-09-14，同素材）：zh「开放时间」AED 与 Qwen 均对；en AED 拼对
"chieftain" 而 Qwen 误听为 "chief then"；粤语 AED 对、Qwen 对；**日文 Qwen 正确、
AED 输出中文乱码**；噪声音频 AED 明显更稳。速度（稳态）AED 约 2.7ms/秒音频（如 43s 音频
≈0.2s），Qwen 约 10-17× 实时；AED 首调有 5-8s 预热。

```bash
python qwen_asr.py 会议录音.m4a                    # 整段转写（自动语言检测；>60s 自动分段提速）
python qwen_asr.py 会议录音.m4a --srt              # 出 .srt 字幕 + 结构化 JSON
python qwen_asr.py 采访.m4a --diarize --srt        # 说话人分离 + 带说话人标签的字幕
python qwen_asr.py 采访.m4a --diarize --names "0=张三,1=李四"   # 说话人改名（先听一段确认谁是谁）
python qwen_asr.py 录音.m4a --replace "小蜜=>小米"  # 专名纠错（可加 --fuzzy 走拼音模糊）
python qwen_asr.py 录音.m4a --engine aed            # 换 FireRedASR2-AED（中英粤更准；日文禁用）
python qwen_asr.py 嘈杂.m4a --denoise              # 含 BGM/强噪前处理
python qwen_asr.py 录音.m4a --vad firered           # 换 FireRedVAD 切段（误报率显著低于 fsmn-vad）
python qwen_asr.py 长录音.m4a --hotwords "热词A,热词B"   # 热词偏置（prompt 层，弱于 --replace）
```

输出：stdout 全文；音频同目录落 `.json`（结构化：segments / sentences）；`--srt` 时额外落 `.srt`。
其余参数：`--language zh` 强制语言、`--threshold 0.7` 聚类阈值、`--merge-gap 800` 同人碎段合并间隔（ms）、
`--max-line 28` 单条字幕最大字数、`--device cpu`、`--no-t2s` 关繁转简、`--replace-file` 用词典文件。

### transcribe.py — FunASR 原生通道（备选）

```bash
python transcribe.py 会议录音.m4a --json out.json
python transcribe.py 录音.wav --device cpu        # 无 N 卡
```

比 Qwen3-ASR 快、显存小，中文准确率略低；作为对照基线与轻量兜底保留。

### diarize.py — 旧版分离脚本（保留，逻辑同 qwen_asr.py --diarize）

```bash
python diarize.py 访谈录音.m4a --threshold 0.7
```

## 重要约束（实测踩坑，勿绕过）

1. **说话人分离走 qwen_asr.py `--diarize` 的手动分环实现**：不要用 AutoModel 的 `spk_model="cam++"` pipeline 参数（funasr 1.4.8 该路径聚类失效，不同人全被标成同一个 spk=0）。
2. **说话人编号 ≠ 真实身份**：0/1/2 按出现顺序分配，交付时提醒用户听一段确认，再用 `--names` 固定。
3. **聚类阈值调节**：两人声音相近被并成一人 → 调低（0.65~0.7）；同一个人被拆成多人 → 调高（0.8）。脚本在簇间相似度贴近阈值时会提示。
4. **`--replace` 优于 `--hotwords`**：热词只是 prompt 偏置（ASR 对 prompt 服从度低，实测连繁简都改不动）；`--replace` 是识别后确定性替换，必中。同音异字用 `--fuzzy`（需 pip 安装 pypinyin+rapidfuzz）。
5. **`--denoise` 是条件性收益**：实测合成白噪声场景**无改善甚至更差**（「开放时间」→「派班时间」）；仅在真实含 BGM/强噪素材上试，干净素材不加任何前处理最好。
6. GPU 上模型输出的 tensor 转 numpy 必须先 `.cpu()`；Windows 原生 Python 不认 Git Bash 的 `/tmp` 路径，临时文件用 `tempfile.gettempdir()`。
7. 字幕时间为字级（fa-zh 强制对齐），比 VAD 段级边界准；单条过长用 `--max-line` 调小。

## 环境自检

用户报告"转写失败/环境不存在"时先跑：

```bash
python -c "import funasr, torch; print(funasr.__version__, torch.cuda.is_available())"
```

- 正常输出 `1.4.x True` → 环境完好，排查音频文件本身（格式/路径）。
- ImportError 或 CUDA 不可用 → 按 references/advanced.md 的「环境安装/修复」节重装（含 RTX 50 系必须 cu128+ 的说明）。

## 详细参考

模型清单与下载 ID、fa-zh 强制对齐 / 专名纠错 / FireRedVAD / ZipEnhancer 的接口与实测数据、ct-punc / emotion2vec 调用代码、环境安装与修复步骤、完整踩坑记录 → 读 **references/advanced.md**（按需加载，不预读）。

## 单元自测

```bash
python scripts/test_qwen_asr_units.py
```
