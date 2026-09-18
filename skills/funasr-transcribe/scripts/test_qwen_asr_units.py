# 自测: qwen_asr.py 的纯函数逻辑(切句/时间戳映射/SRT格式/碎段合并/字幕细分)
# 运行: python scripts/test_qwen_asr_units.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_asr import (split_sentences, char_times, fmt_srt_ts, merge_by_speaker,
                      subtitle_units, PUNCT, apply_replace, build_matcher,
                      aed_ts_usable, aed_aligns_from)

fails = []

def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        fails.append(name)

print("[1] split_sentences 标点切句 + plain 索引")
r = split_sentences("开放时间，早上九点至下午五点。谢谢。", 28)
check("切出 2 句", len(r) == 2, f"got {len(r)}: {r}")
check("句1文本", r[0][2] == "开放时间，早上九点至下午五点。", f"got {r[0][2]!r}")
check("句1 plain 区间", r[0][0] == 0 and r[0][1] == 13, f"got {r[0][:2]}")
check("句2 plain 起点", r[1][0] == 13, f"got {r[1][0]}")
# 无标点整段
r2 = split_sentences("你好世界", 28)
check("无标点=单句", len(r2) == 1 and r2[0][:2] == (0, 4), f"got {r2}")
# 超长句按软标点细分
long_text = "第一段内容" * 5 + "，" + "第二段内容" * 5 + "。"
r3 = split_sentences(long_text, 28)
check("超长句至少切 2 条", len(r3) >= 2, f"got {len(r3)}")
# 标点开头
r4 = split_sentences("。你好。", 28)
check("前置标点不产生空句", len(r4) == 1 and r4[0][2] == "你好。", f"got {r4}")

print("[2] char_times 映射(tokens 数=字符数 / 不匹配兜底)")
ts = [[100, 200], [200, 300], [300, 400], [400, 500]]
ct = char_times("你好世界", ["你", "好", "世", "界"], ts, base_ms=1000, fallback_start=0, fallback_end=5000)
check("4 字符 4 时间戳", ct == [(1100, 1200), (1200, 1300), (1300, 1400), (1400, 1500)], f"got {ct}")
ct2 = char_times("你好，世界", ["你", "好", "世", "界"], ts, base_ms=0, fallback_start=0, fallback_end=4000)
check("标点跳过", len(ct2) == 4 and ct2[0] == (100, 200), f"got {ct2}")
ct3 = char_times("你好世界", ["你", "好"], [[100, 200], [200, 300]], 0, 0, 4000)
check("不匹配兜底均分", len(ct3) == 4 and ct3[0][0] == 0 and ct3[-1][1] == 4000, f"got {ct3}")

print("[3] fmt_srt_ts")
check("零", fmt_srt_ts(0) == "00:00:00,000", fmt_srt_ts(0))
check("1h2m3s456ms", fmt_srt_ts(3723456) == "01:02:03,456", fmt_srt_ts(3723456))
check("负数钳零", fmt_srt_ts(-5) == "00:00:00,000", fmt_srt_ts(-5))

print("[4] merge_by_speaker 相邻同人合并")
segs = [[0, 1000], [1200, 2000], [2500, 3000], [5000, 6000]]
labels = [0, 0, 1, 0]
m = merge_by_speaker(segs, labels, 800)
check("0/0 合并,1 独立,尾 0 独立", len(m) == 3 and m[0]["end_ms"] == 2000 and m[1]["spk"] == 1, f"got {m}")
m2 = merge_by_speaker(segs, labels, 0)
check("gap=0 不合并", len(m2) == 4, f"got {len(m2)}")
m3 = merge_by_speaker([[0, 1000], [1000, 2000]], [0, 0], 0)
check("gap=0 紧邻同人也合", len(m3) == 1, f"got {len(m3)}")

print("[5] subtitle_units 字幕细分(含对齐数据)")
units = [{"start_ms": 0, "end_ms": 4000, "text": "你好世界。再见。", "spk": 0, "spk_name": "张三"}]
aligns = [{"tokens": ["你", "好", "世", "界", "再", "见"], "ts": [[100, 300], [300, 500], [500, 700], [700, 900], [2500, 2700], [2700, 2900]]}]
subs = subtitle_units(units, aligns, 28)
check("切 2 条字幕", len(subs) == 2, f"got {subs}")
check("条1边界贴字级时间戳", subs[0]["start_ms"] == 100 and subs[0]["end_ms"] == 900, f"got {subs[0]}")
check("条2边界", subs[1]["start_ms"] == 2500 and subs[1]["end_ms"] == 2900, f"got {subs[1]}")
check("说话人带名", subs[0]["spk_name"] == "张三", f"got {subs[0]}")
subs2 = subtitle_units(units, None, 28)
check("无对齐=整段一条", len(subs2) == 1 and subs2[0]["start_ms"] == 0, f"got {subs2}")
# 相邻重叠收敛
units_ov = [{"start_ms": 0, "end_ms": 3000, "text": "第一句。第二句。", "spk": None}]
al_ov = [{"tokens": ["第", "一", "句", "第", "二", "句"], "ts": [[0, 1000], [1000, 1200], [1000, 1400], [1300, 1500], [1500, 1700], [1700, 1900]]}]
subs3 = subtitle_units(units_ov, al_ov, 28)
check("重叠收敛(前条 end<=后条 start)", all(subs3[i]["end_ms"] <= subs3[i + 1]["start_ms"] for i in range(len(subs3) - 1)), f"got {subs3}")

print("[6] 专名纠错 matcher")
mm = build_matcher({"开饭时间": "开放时间"}, fuzzy=False)
check("确定性替换", apply_replace("开饭时间早上九点", mm) == "开放时间早上九点", apply_replace("开饭时间早上九点", mm))
check("无 matcher 原样", apply_replace("开饭时间", None) == "开饭时间")
mm2 = build_matcher({}, fuzzy=False)
check("空词典=None", mm2 is None)

print("[7] AED 原生时间戳可用性判定")
ts8 = [("开", 0.1, 0.2)] * 8
check("补标点后仍可用(去标点=8)", aed_ts_usable([{"timestamp": ts8, "transcription": "开放时间，早上九点。"}]) is True)
check("时间戳缺失=不可用", aed_ts_usable([{"timestamp": None, "transcription": "大家好"}]) is False)
check("数量不符=不可用", aed_ts_usable([{"timestamp": ts8[:2], "transcription": "大家好"}]) is False)
al = aed_aligns_from([{"timestamp": [("开", 0.1, 0.25), ("放", 0.25, 0.4)], "transcription": "开放"}])
check("秒->毫秒换算", al[0]["ts"] == [[100, 250], [250, 400]] and al[0]["tokens"] == ["开", "放"], f"got {al}")

print()
if fails:
    print(f"FAILED: {len(fails)} 项 -> {fails}")
    sys.exit(1)
print("ALL PASS")
