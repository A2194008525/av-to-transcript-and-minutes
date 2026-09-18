# 语音转会议纪要（voice-to-minutes）

一套**完全离线、本地运行**的「录音 → 转写 → 会议纪要」全流程工具箱，由两个 AI Agent 技能（Agent Skills / SKILL.md 规范）组成：

```
会议录音 (.m4a/.mp3/.wav/...)
        │
        ▼
┌─────────────────────────────┐
│  funasr-transcribe 语音转写   │  Qwen3-ASR-1.7B + FunASR
│  · 说话人分离（谁在何时说了什么）│  · 字幕 SRT（字级时间戳）
│  · 专名纠错 · 降噪 · 情绪分析  │  · GPU 加速，无需 API 密钥
└─────────────────────────────┘
        │  转写稿（含说话人标签的文本）
        ▼
┌─────────────────────────────┐
│  meeting-notes-expert 纪要整理 │  五段式结构化纪要
│  · 议题去重：一个议题只说一次   │  · 决策标注状态与依据
│  · 待办七列表（含验收标准）    │  · 公文排版规范可直接交付
└─────────────────────────────┘
        │
        ▼
   结构化会议纪要（Markdown / 公文格式）
```

> 当前版本：v1.0 ｜ License: [MIT](LICENSE) ｜ 平台：ZCode / Claude Code 及其他支持 SKILL.md 规范的 AI Agent

---

## ✨ 为什么是这套组合

- **完全离线**：语音识别、声纹聚类、纪要生成全程本地，录音不出机器——会议内容敏感场景可用；无任何 API 密钥。
- **实测踩坑沉淀**：两个技能内嵌 21 条实测踩坑记录（聚类阈值调法、引擎选型对照、PyPI 镜像、显存要求……），不是纸上谈兵。
- **双引擎可切换**：Qwen3-ASR（多语言含日文）/ FireRedASR2-AED（中英粤更准），一条参数切换。
- **从原始录音到可执行纪要**：转写稿自动带说话人标签与时间戳，纪要环节按议题聚合去重，待办带责任人与验收标准。

## 📦 仓库结构

```
voice-to-minutes/
├── README.md            ← 本文件（全流程总览）
├── LICENSE              ← MIT
└── skills/
    ├── funasr-transcribe/        ← 技能一：本地语音转写
    │   ├── SKILL.md              ← 技能定义（安装时只需要这个目录）
    │   ├── references/
    │   │   └── advanced.md       ← 模型清单/环境安装/21 条踩坑记录
    │   └── scripts/
    │       ├── qwen_asr.py       ← 主力脚本（双引擎/分离/字幕/纠错）
    │       ├── transcribe.py     ← FunASR 原生通道（备选）
    │       ├── diarize.py        ← 旧版分离脚本（保留）
    │       └── test_qwen_asr_units.py  ← 纯函数单元自测
    └── meeting-notes-expert/
        └── SKILL.md              ← 技能二：会议纪要整理
```

## 🚀 安装

### 1）环境准备（转写技能需要，一次性）

- Python 3.10+，NVIDIA 显卡（RTX 50 系必须 cu128+ 的 torch；无 N 卡可用 `--device cpu`，速度慢）

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install funasr modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

模型首次运行自动下载（主力 Qwen3-ASR-1.7B 约 4.1GB），缓存在 `~/.cache/modelscope/models/`。

### 2）安装两个技能

```bash
git clone https://github.com/A2194008525/voice-to-minutes.git
```

把 `skills/` 下两个文件夹整体拷入你的 Agent 技能目录：

| 平台 | 技能目录 |
| --- | --- |
| ZCode | `~/.zcode/skills/` |
| Claude Code | `~/.claude/skills/` |

### 3）自检

```bash
python skills/funasr-transcribe/scripts/test_qwen_asr_units.py   # 应输出 ALL PASS
python -c "import funasr, torch; print(funasr.__version__, torch.cuda.is_available())"
```

## 💬 使用

装好后对 Agent 说人话即可，两条命令走完全流程：

```text
第 1 步：把这段会议录音转写一下，要区分说话人 → audio/20260918周会.m4a
第 2 步：把转写稿整理成会议纪要 → 20260918周会_transcript 的内容
```

也可以直接给单个技能派活：

| 场景 | 示例指令 |
| --- | --- |
| 只要转写 | `python skills/funasr-transcribe/scripts/qwen_asr.py 会议录音.m4a --diarize --srt` |
| 只要纪要 | 「把这段会议记录整理成纪要」（直接给文本） |
| 字幕制作 | 「给这个视频的录音出一份 SRT 字幕」 |

> 转写完成后输出的文本（stdout 或 `.json`）直接交给 meeting-notes-expert 即可；两个技能也可独立使用。

## 📋 两个技能各自的能力

### funasr-transcribe（语音 → 文字）

整段转写、说话人分离（VAD 切段 → CAM++ 声纹 → 余弦层次聚类的手动分环实现）、SRT 字幕（fa-zh 强制对齐出字级时间戳）、专名确定性纠错（`--replace`）、拼音模糊纠错（`--fuzzy`）、可选降噪（ZipEnhancer）、录音情绪分析（emotion2vec）、繁转简。引擎选型与参数详见 [skills/funasr-transcribe/SKILL.md](skills/funasr-transcribe/SKILL.md)。

### meeting-notes-expert（文字 → 纪要）

五段式纪要（基本信息 / 会议内容 / 核心要点 / 会议总结 / 待办表格）；「一个议题只说一次」去重红线；关键决策标注状态（已定/待定/待确认）与可溯源依据；待办七列表（含验收标准），按优先级→时间排序；内置公文排版规范（A4、宋体/仿宋分级字号、Word 导航标题）。详见 [skills/meeting-notes-expert/SKILL.md](skills/meeting-notes-expert/SKILL.md)。

## ⚠️ 已知限制

- 说话人分离依赖声纹聚类，**说话人编号 ≠ 真实身份**（按出现顺序分配），交付前需人工听一段确认，再用 `--names` 固定。
- FireRedASR2-AED 引擎不支持日文；中日混合场景用默认 Qwen 引擎。
- 降噪（`--denoise`）是条件性收益：实测对白噪声无改善甚至更差，仅真实 BGM/强噪素材建议试用。
- 首次运行需下载模型（主力约 4.1GB），请保持网络可用；之后完全离线。

## 🙏 致谢

- [Qwen3-ASR](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B-hf) 与 [FunASR](https://github.com/modelscope/FunASR)（阿里达摩院）——识别与 VAD/声纹/对齐基座
- [FireRedASR2](https://github.com/FireRedTeam/FireRedASR2S)（小红书）——可选高准确率引擎
- [mohui233/meeting-minutes](https://github.com/mohui233/meeting-minutes)——纪要技能 v1.1 合入的三点写作纪律参考

## 📄 许可

本项目采用 [MIT License](LICENSE) 开源：可自由使用、修改、再分发（含商用），只需保留原版权与许可声明；软件按「现状」提供，不含任何担保。所依赖的模型各自遵循其上游许可。
