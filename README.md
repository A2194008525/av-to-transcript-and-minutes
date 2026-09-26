# 音视频 → 转写文稿 / 会议纪要（av-to-transcript-and-minutes）

一套**完全离线、本地运行**的「音视频 → 转写文稿 / 会议纪要」全流程工具箱：

```
视频 / 音频（mp4/mkv/mp3/m4a/wav/…）
        │
        ├─ ① 拆轨（视频输入）        无声视频 + 完整音轨（无损，FFmpeg）
        │
        ├─ ② 人声 / 背景音分离        人声.wav + 背景音.wav（Demucs htdemucs，GPU）
        │
        ├─ ③ 噪声门清理              人声_转写用.wav（消除分离模型的静音伪影，保 VAD 可用）
        │
        ├─ ④ 说话人分离 + 转写        Qwen3-ASR-1.7B × (pyannote / cam++)，带时间戳与说话人
        │
        └─ ⑤ 组稿 + 排版规范 docx    转写文稿（普通）  或  会议原文（带标签，供提炼纪要）
                                     正式纪要由 AI Agent 按 meeting-notes-expert 模板提炼
```

**一条命令跑完 ①→⑤**：

```bash
python av_to_transcript_and_minutes.py <视频/音频文件或文件夹> [-o 输出目录] [--meeting] [--srt] …
```

> 当前版本：**v2.1** ｜ License: [MIT](LICENSE) ｜ 环境：Windows / Linux / macOS，需 FFmpeg 与 Python 3.10+

---

## ✨ 特性

- **完全离线**：模型全部本地运行，无 API 密钥、无网络依赖（首次运行自动下载模型后）
- **一条命令全自动**：拆轨 → 分离 → 说话人分离 → 转写 → 排版成 docx
- **双产出模式**：`转写文稿-<名>.docx`（普通音视频）／`会议原文-<名>.docx`（会议，带 `[分:秒] 说话人N：` 标签）
- **说话人分离双引擎**：
  - `pyannote`（community-1，会议场景默认）——分割模型原生处理**重叠语音**，实测真实 28 分钟会议从 35 个假说话人收敛到 2 个
  - `campp`（VAD + cam++ 声纹聚类）——轻量，单人/访谈素材够用
- **字幕与纠错**：`--srt` 出字级时间戳字幕；`--hotwords` 热词偏置；`--names` 说话人真名映射
- **断点续跑**：转写（最贵环节）结果按参数指纹缓存，中断重跑自动跳过
- **批量处理**：输入文件夹时递归处理其中所有音视频

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 系统依赖：FFmpeg（须在 PATH 中）
winget install ffmpeg            # Windows
# apt install ffmpeg            # Linux
# brew install ffmpeg           # macOS

# Python 依赖
pip install -r requirements.txt
# GPU 加速：先按 https://pytorch.org 安装对应 CUDA 版本的 torch

# 若报 libtorchcodec 加载失败（pyannote 引入的库与静态 FFmpeg 不兼容）：
pip uninstall torchcodec
```

### 2. 模型准备（首次，自动下载）

| 用途 | 模型 | 来源 |
| --- | --- | --- |
| 转写识别 | Qwen3-ASR-1.7B | ModelScope `Qwen/Qwen3-ASR-1.7B-hf` |
| 说话人分离（会议） | pyannote community-1 | ModelScope `pyannote/speaker-diarization-community-1` |
| 说话人分离（轻量） | fsmn-vad + cam++ | 随 FunASR 自动下载 |
| 人声分离 | htdemucs | 随 Demucs 自动下载 |

> pyannote 官方模型在 HuggingFace 为门控资源（需申请权限）；**ModelScope 有官方镜像且无需门控**，推荐：
> ```bash
> python -c "from modelscope import snapshot_download; snapshot_download('pyannote/speaker-diarization-community-1')"
> ```

### 3. 跑起来

```bash
# 普通音视频 → 转写文稿（自动说话人分离）
python av_to_transcript_and_minutes.py 我的视频.mp4

# 会议录音 → 会议原文（带时间戳与说话人标签，供后续提炼纪要）
python av_to_transcript_and_minutes.py 会议录音.m4a --meeting

# 批量 + 字幕 + 人名映射 + 热词
python av_to_transcript_and_minutes.py ./素材目录 --meeting --srt \
    --names "0=张三,1=李四" --hotwords "产品名,行业术语"
```

### 4. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--meeting` | 产出会议原文（带时间戳+说话人标签）而非普通转写文稿 |
| `--srt` | 额外产出 `.srt` 字幕（字级时间戳） |
| `--diarize-engine auto\|pyannote\|campp` | 说话人分离引擎（`auto`=会议用 pyannote，其余 campp） |
| `--names "0=张三,1=李四"` | 说话人真名映射 |
| `--hotwords "词1,词2"` | 热词偏置，提升专名识别率 |
| `--no-separate` | 跳过人声分离（纯人声音频可直接转写） |
| `--no-asr` | 只做拆轨/分离，不转写 |
| `--force` | 重跑已处理过的素材 |
| `-o 输出目录` | 指定输出目录（默认输入旁 `separated_out/`） |

---

## 📦 产物说明

每个输入在输出目录下生成同名子目录：

| 文件 | 说明 |
| --- | --- |
| `无声视频.mp4` | 去掉声音的视频（仅视频输入，画面无损） |
| `完整音轨.m4a` | 从视频无损提取的完整音频（仅视频输入） |
| `人声.wav` / `背景音.wav` | Demucs 分离出的人声与背景音 |
| `人声_转写用.wav` | 经噪声门清理后供转写的版本（可复现中间件） |
| `转写结果*.json` | 结构化转写结果（说话人分段/文本/时间戳），按参数指纹缓存 |
| `转写文稿-<名>.md/.docx` | 普通模式成稿（如有 article-format 技能则套用排版规范） |
| `会议原文-<名>.md/.docx` | 会议模式成稿（`[00:00] 说话人N：文本`） |
| `会议原文-<名>.srt` | 字幕（`--srt`，含说话人标记） |

**会议纪要**由 AI Agent 读取「会议原文」后按 [meeting-notes-expert](skills/meeting-notes-expert/SKILL.md) 模板提炼成五段式纪要（基本信息 / 会议内容 / 核心要点 / 会议总结 / 待办事项表）——这一步是理解性工作，交给会话内的 AI 完成，产出 `会议纪要-<名>.docx`。

---

## 🧩 项目结构

```
av-to-transcript-and-minutes/
├── av_to_transcript_and_minutes.py  # 全自动链路主脚本（①→⑤）
├── test_pipeline_e2e.py             # 端到端回归（自动合成素材，18 项断言）
├── requirements.txt
├── skills/
│   ├── funasr-transcribe/
│   │   ├── SKILL.md                 # 转写技能（Agent 调用规范）
│   │   └── scripts/
│   │       ├── qwen_asr.py          # 转写引擎：Qwen3-ASR + pyannote/cam++ + 字幕/纠错
│   │       ├── transcribe.py        # 简化入口
│   │       ├── diarize.py           # 说话人分离（FunASR 路线）
│   │       └── test_qwen_asr_units.py  # 引擎单元测试
│   └── meeting-notes-expert/
│       └── SKILL.md                 # 纪要整理技能（五段式模板）
└── LICENSE
```

---

## ✅ 验证

```bash
# 端到端回归（自动合成 TTS 测试素材，断言全链路产物）
python test_pipeline_e2e.py

# 引擎单元测试（切句/时间戳/碎段合并/超长切分/小簇吸收等纯逻辑）
python skills/funasr-transcribe/scripts/test_qwen_asr_units.py
```

实测性能参考（RTX 5070 Ti 16GB，28 分钟中文会议）：

| 环节 | 耗时 |
| --- | --- |
| 人声分离（Demucs） | ~1 分钟 |
| 说话人分离（pyannote） | ~40 秒 |
| 转写（Qwen3-ASR） | ~87 秒 |
| **全流程** | **约 4 分钟** |

---

## 🔧 技术要点与已知限制

**关键设计**

- **噪声门（必需）**：Demucs 在静音段输出低电平伪影，会让下游 VAD 失效（整段连成一块、说话人分离报废）。链路在分离与转写之间插入 `agate` 噪声门恢复段间真静音。
- **三层声纹防护**（cam++ 引擎）：碎段合并（<400ms）→ 超长段切分（>60s，保批量转写吞吐）→ 小簇吸收（<5s 的碎簇并入最相似大簇）。
- **pyannote 优先**：实测真实会议抢话交叠场景，pyannote 分离质量显著优于声纹聚类路线。
- **torchcodec 规避**：pyannote 引入的 torchcodec 与静态 FFmpeg 不兼容；链路用 soundfile 预载波形绕过，建议 `pip uninstall torchcodec`。
- **AED 显存三层防护**（v2.1）：FireRedASR2-AED 引擎经 `firered_mem_patch.py` 分块注意力（softmax 按行独立，数学等价）+ 时长感知分批（`AED_BATCH_BUDGET_S`）+ OOM 拆批/静音点二分兜底，再配 bf16（`use_half=True`）与 decoder 三角 mask 缓存；实测 4×107s 批量峰值显存 11.8GB → 3.5GB、推理提速 3 倍（不改官方源码，monkey-patch 可还原）。

**已知限制**

- 会议**抢话极严重**时说话人分离仍可能过切（实测 35 簇 → 2 簇已大幅改善，但不保证 100% 准确）
- 说话人编号在不同次运行间可能互换（可用 `--names` 固定映射）
- FireRedASR2-AED 为可选备选引擎，需另行下载模型与代码（v2.1 起长音频已优化，见上）
- 输出 docx 的排版规范依赖 `article-format` 技能（未安装时产出标准 docx，无规范排版）

---

## 📄 License

[MIT](LICENSE)
