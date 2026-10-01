#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""itn_zh 单元测试：正例（该转的要转对）+ 反例（不该转的一个都不许动）。

运行：python scripts/test_itn_zh.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itn_zh import cn_to_int, normalize  # noqa: E402

fails = []


def check(name, got, want):
    if got == want:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name}: got={got!r} want={want!r}")
        fails.append(name)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("[1] cn_to_int 读数")
    check("三百二十", cn_to_int("三百二十"), 320)
    check("十五", cn_to_int("十五"), 15)
    check("一万二千", cn_to_int("一万二千"), 12000)
    check("十", cn_to_int("十"), 10)
    check("三", cn_to_int("三"), 3)
    check("纯逐位长度>1 默认不转", cn_to_int("三五"), None)
    check("逐位允许时『二零二六』", cn_to_int("二零二六", allow_digitwise=True), 2026)
    check("非法字返回 None", cn_to_int("三X"), None)

    print("[2] 百分比")
    check("百分之八十", normalize("百分之八十"), "80%")
    check("百分之十五点五", normalize("百分之十五点五"), "15.5%")
    check("千分之三", normalize("千分之三"), "3‰")
    check("百分之百", normalize("百分之百"), "100%")

    print("[3] 序数与年份日期")
    check("第三百二十台", normalize("第三百二十台"), "第320台")
    check("第二季度", normalize("第二季度"), "第2季度")
    check("二零二六年", normalize("二零二六年"), "2026年")
    check("日期", normalize("二零二六年十月十五日"), "2026年10月15日")
    check("日号", normalize("十月十五号"), "10月15号")

    print("[4] 数量 + 单位")
    check("三百二十万元", normalize("三百二十万元"), "320万元")
    check("五分钟", normalize("五分钟"), "5分钟")
    check("十五个", normalize("十五个"), "15个")
    check("三十分钟", normalize("三十分钟"), "30分钟")
    check("一万二千元", normalize("一万二千元"), "12000元")
    check("十台设备", normalize("十台设备"), "10台设备")
    check("三个人", normalize("三个人"), "3个人")
    check("一个字", normalize("一个字"), "1个字")

    print("[5] 反例：非数量用法必须原样保留")
    check("一起开会", normalize("一起开会"), "一起开会")
    check("十分重要", normalize("十分重要"), "十分重要")
    check("三天打鱼", normalize("三天打鱼"), "三天打鱼")
    check("三五个人", normalize("三五个人"), "三五个人")
    check("一样", normalize("一样"), "一样")
    check("一方面", normalize("一方面"), "一方面")
    check("一系列", normalize("一系列"), "一系列")
    check("一致同意", normalize("一致同意"), "一致同意")
    check("第一次见面里的『见面』不受影响", normalize("第一次见面"), "第1次见面")

    print("[6] 混合句")
    src = "第三季度的采购预算总共是三百二十万元，请财务在二零二六年十月十五日之前完成审批，占比约百分之十五点五。"
    want = "第3季度的采购预算总共是320万元，请财务在2026年10月15日之前完成审批，占比约15.5%。"
    check("整句混合", normalize(src), want)

    print("[7] 边界：空串与非数字文本")
    check("空串", normalize(""), "")
    check("无数字", normalize("今天天气不错"), "今天天气不错")

    print()
    if fails:
        print(f"FAILED: {len(fails)} 项 -> {fails}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
