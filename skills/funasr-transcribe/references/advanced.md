# advanced.md — 模型清单、附加能力、环境修复、完整踩坑

按需加载,SKILL.md 正文未覆盖的细节都在这里。

## 一、模型清单（全离线,默认缓存于 ~/.cache/modelscope/models/）

| 模型 | ModelScope ID | 大小 | 用途 |
|------|--------------|------|------|
| **Qwen3-ASR-1.7B-hf（主力识别·默认）** | `Qwen/Qwen3-ASR-1.7B-hf` | 4.08GB | 中英日粤+多语言,自带标点与语种识别,transformers 原生直载 |
| **FireRedASR2-AED（可选引擎,中英粤更准）** | `xukaituo/FireRedASR2-AED` | 4.73GB | 字级时间戳+置信度原生;不支持日文;输出无标点(需 ct-punc);只吃 16k wav |
| FireRedASR2S 代码 | GitHub `FireRedTeam/FireRedASR2S` | 17MB | AED 引擎的推理代码,落 `~\.local\fire-red-asr2s`（脚本自动加 sys.path） |
| **fa-zh（强制对齐）** | `iic/speech_timestamp_prediction-v1-16k-offline` | 38M 参数 | 字级时间戳:输入(音频,文本)出 [[start_ms,end_ms],...],字幕制作靠它 |
| Fun-ASR-Nano-2512(备选) | `FunAudioLLM/Fun-ASR-Nano-2512` | 2.0GB | 中/英/日+方言,Llama 架构 ASR,自带标点与时间戳,快且省显存 |
| fsmn-vad | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`(别名 fsmn-vad) | 3.9MB | 语音段切分(误报率偏高,见 FireRedVAD) |
| cam++ | `iic/speech_campplus_sv_zh-cn_16k-common`(别名 cam++) | 28MB | 192 维声纹 embedding |
| ct-punc | `iic/punc_ct-transformer_cn-en-common-vocab471067-large`(别名 ct-punc) | 1.2GB | 中英标点恢复 |
| emotion2vec+large | `iic/emotion2vec_plus_large` | ~1GB | 8 类情绪识别 |
| seaco-paraformer | `iic/speech_seaco_paraformer_large_...`(别名 paraformer-zh) | 953MB | 备选中文 ASR(支持热词,英文弱) |
| **FireRedVAD** | `xukaituo/FireRedVAD`（缓存 `~\.cache\fireredvad\FireRedVAD\VAD`） | 2.2MB | VAD 升级件:误报率 2.69% vs fsmn-vad 44.03%（官方自评） |
| **ZipEnhancer（降噪）** | `iic/speech_zipenhancer_ans_multiloss_16k_base` | ~2M 参数 | 16k 语音增强,含 BGM/强噪时可选前处理 |

funasr AutoModel 的短别名(fsmn-vad/cam++/ct-punc/paraformer-zh/fa-zh)可直接用,首次解析会自动下载。

## 一点五、fa-zh 强制对齐（字级时间戳 / 字幕）

`qwen_asr.py --srt` 内部流程；单独调用：

```python
from funasr import AutoModel
fa = AutoModel(model="fa-zh", device="cuda:0", disable_update=True)
r = fa.generate(input=("录音.wav", "要对的文字"), data_type=("sound", "text"))
# r[0]["text"] = "字 符 序 列"（空格分隔 token）
# r[0]["timestamp"] = [[790,1030], [1050,1290], ...]  单位毫秒，与 token 一一对应
# 批量：input=([wav1,wav2],[text1,text2]) 同样形状
```

实测（2026-09-14）：标点自动跳过（13 汉字+标点 → 13 token/13 时间戳）；83.8s 音频单次对齐
11 段全部成功、时间戳单调；长句可直接整段进（无 60s 限制）。

## 一点六、专名纠错（postprocess_hotwords，确定性替换）

比 `--hotwords`（prompt 偏置，服从度低）可靠；`qwen_asr.py --replace / --replace-file / --fuzzy` 封装：

```python
from funasr.utils.postprocess_hotwords import build_postprocess_hotword_matcher
m = build_postprocess_hotword_matcher(postprocess_hotwords={"开饭时间": "开放时间"}, enable_fuzzy=False)
new_text, matches = m.apply_text("开饭时间早上九点至下午五点。")
# new_text = "开放时间早上九点至下午五点。"
```

词典文件格式：每行 `错=>对`（确定性）或只写目标词（拼音模糊，需 `enable_fuzzy=True` + pypinyin + rapidfuzz）。

## 二、ct-punc 标点恢复(给外部文本补标点)

Fun-ASR-Nano 转写输出自带标点,ct-punc 主要用于**无标点的转写稿/外部文本**:

```python
from funasr import AutoModel
punc = AutoModel(model="ct-punc")
res = punc.generate(input="我今天来参加这个会议主要是想讨论下季度的产品规划另外还有两件事需要确认")
print(res[0]["text"])  # 我今天来参加这个会议，主要是想讨论下季度的产品规划。另外还有两件事需要确认。
```

## 三、emotion2vec+large 录音情绪分析

```python
from funasr import AutoModel
emo = AutoModel(model="iic/emotion2vec_plus_large", device="cuda:0")
r = emo.generate(input="录音.wav", cache={}, granularity="utterance")
top = sorted(zip(r[0]["labels"], r[0]["scores"]), key=lambda x: -x[1])[:3]
# 输出形如 [('中立/neutral', 0.98), ('开心/happy', 0.01), ...],标签含中文
```

8 类情绪:开心/难过/厌恶/中立/生气/惊讶/害怕/兴奋(以实际标签为准)。granularity="utterance" 按整段分析。

## 四、热词定制(Fun-ASR-Nano 与 paraformer-zh 均支持)

专有名词/人名转不准时,generate 加 `hotwords=["专有名词A", "专有名词B"]` 提升命中。diarize.py/transcribe.py 未暴露此参数,需要时在脚本 generate 调用处加一行。

## 四点五、FireRedVAD / ZipEnhancer 用法

```python
# FireRedVAD（qwen_asr.py --vad firered 已封装）
from pathlib import Path
from fireredvad import FireRedVad, FireRedVadConfig
cfg = FireRedVadConfig(use_gpu=True)
vad = FireRedVad.from_pretrained(str(Path.home() / ".cache" / "fireredvad" / "FireRedVAD" / "VAD"), cfg)
result, probs = vad.detect("录音.wav")   # 输入须 16k 单声道 wav
# result = {"dur": 66.2, "timestamps": [(0.66, 5.36), ...]}  单位=秒
```

```python
# ZipEnhancer 降噪（qwen_asr.py --denoise 已封装）
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ans = pipeline(Tasks.acoustic_noise_suppression, model="iic/speech_zipenhancer_ans_multiloss_16k_base")
ans("嘈杂.wav", output_path="enhanced.wav")   # 16k 输入，返回 output_pcm 并落盘
```

实测（2026-09-14）：FireRedVAD 对 66.2s 音频 0.13s 切出 10 段（fsmn-vad 同音频 10 段，
但时间边界更贴）｜ZipEnhancer 5.6s 音频处理 10.1s；**白噪声场景降噪后识别反而变差**
（「开放时间」→「派班时间」），仅建议真实 BGM/强噪素材试用。

## 五、环境安装/修复

前置:Python 3.10+（3.12 实测通过）。三步:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130   # RTX 50系(Blackwell sm_120)必须 cu128+;cu130 与驱动 UMD 13.4 匹配
pip install funasr modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 30   # 官方 PyPI 源实测有长时间卡死案例,默认用清华镜像
python -c "import funasr, torch; print(funasr.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

验证转写(模型包自带示例音频,首跑会下载主模型):

```bash
python <技能目录>/scripts/transcribe.py "<modelscope缓存目录>/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master/example/zh.mp3"
# 默认缓存目录: ~/.cache/modelscope/models/（Windows 即 %USERPROFILE%\.cache\modelscope\models\）
```

预期输出含「开饭时间早上九点至下午五点。」。热启动加载约 20s,转写秒级。

无 N 卡/驱动异常时:脚本全部支持 `--device cpu`(diarize.py)或改 device 参数;transcribe.py 需改脚本内 device="cuda:0" 为 "cpu"。

## 六、完整踩坑记录(2026-08-30 实测)

1. **spk_model pipeline 失效**:AutoModel 挂 `spk_model="cam++"` 后 spk 恒为 0(Fun-ASR-Nano 与 paraformer-zh 双路线复现)。根因疑似 pipeline 对整条音频只提一次声纹、所有段共享。cam++ 单独调用是好的(两人余弦相似度 0.034)。→ 说话人分离只能走 diarize.py/qwen_asr.py 的手动分环。
2. **README 未勾选的能力不可信**:Fun-ASR-Nano README「支持区分说话人识别」是未勾选 checkbox,与实测一致。选型前先看模型 README 的 checkbox 状态,再实测金标准。
3. **Blackwell 显卡装 torch**:RTX 50 系 sm_120 在 cu121/cu124 wheel 上报 no kernel image;必须 2.7.0+cu128 起的 wheel。
4. **官方 PyPI 源卡死**:实测 `pip install funasr modelscope` 官方源 28 分钟零进展(缓存零增长确诊卡死);换清华镜像 45 秒装完。判卡死:间隔 30s 对比 pip 缓存目录大小。
5. **GPU tensor→numpy**:必须 `x.cpu().numpy()`;直接 np.array(CUDA tensor) 报错。
6. **Windows 原生 Python 不认 Git Bash `/tmp`**:临时文件一律 `tempfile.gettempdir()`。
7. **funasr 批量输入限制**:generate(input=[numpy, numpy]) 不被 Fun-ASR-Nano 支持(NoneType);传文件路径列表可用。diarize.py 据此逐段写临时 wav。
8. **英文/多语言选型**:paraformer-zh 英文翻车实例("chieftain"→"drible tifton");中英混合一律 Fun-ASR-Nano 或 Qwen3-ASR。
9. **验证素材**:模型包 example/ 下 zh/en/ja/ko/yue.mp3 可做测试;拼接两个不同语言音频 + 1s 静音即可构造双说话人测试素材。

### 2026-09-14 新增（Qwen3-ASR 混搭升级轮）

10. **transformers 音频参数必须传列表**:`apply_transcription_request(audio=[path])` 用 list；传单个 `Path` 对象报 `Invalid input type`（脚本内统一 `str` + list）。
11. **fa-zh 输入契约**:`input=(wav, text)` 或 `([wavs],[texts])` + `data_type=("sound","text")`；传 `[(wav,text)]`（元组列表）会让 VAD 系报 broadcast shape 错（那是另一条管线）。文本里的标点会被自动跳过,不用预处理。
12. **ZipEnhancer 白噪声无收益**:见 §四点五 实测；降噪是条件性收益,不是默认项。
13. **FireRedVAD 装法**:`pip install fireredvad`（纯 Python wheel,依赖 kaldi-native-fbank 有 cp312 预编译轮,零原生编译）；**别用它源码仓库的 requirements.txt**（钉死 torch 2.1.0+cu118 会砸环境）。
14. **VAD 输出单位不同**:fsmn-vad 返回毫秒 `[[s,e],...]`;FireRedVAD 返回秒 `[(s,e),...]`——混用时必须换算（qwen_asr.py 已统一成毫秒）。

### FireRedASR2-AED 引擎（2026-09-14 接入）

15. **只吃 16k 单声道 wav**:内部 `kaldiio.load_mat` 读音频,mp3 直接报 `read_ascii_mat` 解码错。脚本已自动转（`--engine aed` 时先转 wav）。
16. **不支持日文**:实测日文音频输出中文乱码;中日混用场景用默认 Qwen 引擎。
17. **输出无标点**:AED 只出纯字符,脚本默认调 ct-punc 补（`--no-punc` 关闭）。**补标点后文本长度 ≠ 时间戳数**,脚本按"去标点字符数"比对,数量一致才用原生时间戳,否则回落 fa-zh。
18. **60s 上限的实测实情**:README 称 60s,实测 93s 音频正常（130→182 字符输出、置信 0.997）,但官方未承诺,超 60s 建议加 `--srt`/分段跑以稳。
19. **配置参数名要照抄官方示例**:`FireRedAsr2Config` 无 `sentencepiece_model` 参数（脚本曾多传致 TypeError）。
20. **装法**:模型 `modelscope download --model xukaituo/FireRedASR2-AED`（4.73GB）；代码 `git clone FireRedTeam/FireRedASR2S`（约 17MB,Apache-2.0）；依赖缺 `cn2an`（`pip install cn2an`）,kaldi_native_fbank/kaldiio 已在 FireRedVAD 一轮装好。**别用其 requirements.txt**（钉死 torch 2.1.0+cu118）。
21. **显存**:AED 模型权重约 4.4GB（fp32 1152M 参数）,16GB 卡可全载;与 Qwen3-ASR 不共存时无需卸载。
