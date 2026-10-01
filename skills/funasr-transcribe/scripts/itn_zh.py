#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文逆文本正则化（ITN）：把口语化的数字形态转成书面形态，零第三方依赖。

为什么自写而不是用现成实现：官方两条路都走不通——
  ① WeTextProcessing 依赖 pynini，而 pynini 在 PyPI 只有源码包，Windows 下需要
     MSVC Build Tools 编译（本机实测 `Failed building wheel for pynini`，且无 cl.exe）；
  ② fun_text_processing 不在 PyPI（`No matching distribution found`）。

覆盖（保守策略，只转高置信模式）：
  百分之X / 千分之X      → X% / X‰（支持「百分之十五点五」这类小数）
  第X（序数）            → 第X
  四字年份（逐位读法）    → 二零二六年 → 2026年
  月 / 日 / 号           → 十月十五日 → 10月15日
  数量 + 安全量词/单位    → 三百二十万元 → 320万元；五分钟 → 5分钟

刻意不转（避免误伤）：
  「一起 / 一定 / 一样 / 十分 / 三天打鱼 / 三五个」这类非数量或约数用法。
  判据两条：①量词必须落在安全表内（含「天」「分」等高歧义字的一律不入表）；
  ②非百分号场景只接受常规读法——纯逐位串且长度大于 1 的一律不转
  （「三五」不转，「十五」「三百」「二零二六」照转，年份是逐位读法的唯一例外）。

已知边界：不做分词、不查词典，因此「三五个」「七八成」这类约数保持原样而不是猜错；
「一个劲」「一条心」这类惯用语存在误伤可能。
"""

import re

_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_BIG = {"万": 10000, "亿": 100000000}

# 安全量词/单位表：只放“后面跟这个字基本就是数量”的词。
# 刻意排除：天（三天打鱼）、分（十分好）、点（一点）、些/般/样/致/起/直/定/切/共/边/旦/向…
_SAFE_UNITS = [
    "万元", "亿元", "千瓦时", "摄氏度", "平方米", "立方米",
    "公斤", "千米", "毫米", "毫升", "分钟", "小时", "季度",
    "元", "台", "部", "辆", "架", "艘", "只", "件", "套", "张", "份", "条", "名", "位",
    "家", "间", "层", "楼", "页", "行", "倍", "折", "成", "笔", "项", "款", "个", "次",
    "人", "岁", "号", "日", "月", "年", "克", "吨", "升", "米", "度", "周", "秒",
    "字", "句", "段", "章", "节", "课", "班", "组", "队", "户", "所", "座", "门",
]

_CN_NUM = "[零〇一二三四五六七八九十百千万亿两]+"
_CN_NUM_LAZY = "[零〇一二三四五六七八九十百千万亿两]+?"
_UNIT_RE = "|".join(sorted(set(_SAFE_UNITS), key=len, reverse=True))
_YEAR_CHARS = "[零〇一二三四五六七八九]"


def _cn_to_int_conventional(text):
    """常规读法：三百二十 → 320；十五 → 15；一万二千 → 12000"""
    total, section, num = 0, 0, 0
    for ch in text:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            section += (num if num else 1) * _CN_UNIT[ch]
            num = 0
        elif ch in _CN_BIG:
            section = (section + num) * _CN_BIG[ch]
            total += section
            section, num = 0, 0
        else:
            return None
    return total + section + num


def cn_to_int(text, allow_digitwise=False):
    """中文数字 → int。

    allow_digitwise=False（默认，用于数量短语）：纯逐位串且长度 >1 时返回 None，
      这样「三五个人」不会被转成「35个人」。
    allow_digitwise=True（用于年份）：二零二六 → 2026。
    """
    if not text:
        return None
    if any(ch in _CN_UNIT or ch in _CN_BIG for ch in text):
        # 「万」「亿」这类量级字单独出现时不构成数值。否则已转好的「320万元」会被
        # 「万」+「元」二次匹配，实测把它改成了「3200元」（cn_to_int("万") 返回 0）。
        if all(ch in _CN_BIG for ch in text):
            return None
        return _cn_to_int_conventional(text)
    if len(text) > 1 and not allow_digitwise:
        return None
    val = 0
    for ch in text:
        d = _CN_DIGIT.get(ch)
        if d is None:
            return None
        val = val * 10 + d
    return val


def _sub_pct(match):
    base = 100 if match.group("kind") == "百" else 1000
    val = cn_to_int(match.group("int_part"), allow_digitwise=True)
    if val is None:
        return match.group(0)
    frac = match.group("frac")
    if frac:
        frac_val = cn_to_int(frac, allow_digitwise=True)
        if frac_val is None:
            return match.group(0)
        return f"{val}.{frac_val}{'%' if base == 100 else '‰'}"
    return f"{val}{'%' if base == 100 else '‰'}"


def _sub_ordinal(match):
    val = cn_to_int(match.group(1), allow_digitwise=True)
    return f"第{val}" if val is not None else match.group(0)


def _sub_year(match):
    val = cn_to_int(match.group(1), allow_digitwise=True)
    return f"{val}年" if val is not None else match.group(0)


def _sub_num_unit(match):
    val = cn_to_int(match.group(1))
    return f"{val}{match.group(2)}" if val is not None else match.group(0)


_PCT_RE = re.compile(
    r"(?P<kind>[百千])分之(?P<int_part>" + _CN_NUM + r")"
    r"(?:点(?P<frac>" + _CN_NUM + r"))?")
_ORDINAL_RE = re.compile(r"第(" + _CN_NUM + r")")
_YEAR_RE = re.compile(r"(" + _YEAR_CHARS + r"{4})年")
_MONTH_RE = re.compile(r"(" + _CN_NUM + r")(月)")
_DAY_RE = re.compile(r"(" + _CN_NUM + r")([日号])")
_WAN_RE = re.compile(r"(" + _CN_NUM_LAZY + r")(万元|亿元)")
_NUM_UNIT_RE = re.compile(r"(" + _CN_NUM_LAZY + r")(" + _UNIT_RE + r")")


def normalize(text):
    """对一段文本做 ITN；不匹配的地方原样保留。text 为空返回空。"""
    if not text:
        return text
    out = _PCT_RE.sub(_sub_pct, text)
    out = _ORDINAL_RE.sub(_sub_ordinal, out)   # 「第」后必是序数，最安全
    out = _YEAR_RE.sub(_sub_year, out)         # 四位年份走逐位读法
    out = _MONTH_RE.sub(_sub_num_unit, out)
    out = _DAY_RE.sub(_sub_num_unit, out)
    out = _WAN_RE.sub(_sub_num_unit, out)      # 万元/亿元优先于「元」
    out = _NUM_UNIT_RE.sub(_sub_num_unit, out)
    return out


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for line in (sys.argv[1:] or [sys.stdin.read()]):
        print(normalize(line))
