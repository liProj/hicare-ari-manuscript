"""Transcribe the numbers printed in Medicina 62:1752 from the article's PDF text.

Outputs (results/paper/):
  table1.csv   every cell of Table 1 (counts by outcome, sample and level)
  or_tables.csv  Tables 5-7: crude OR and adjusted OR with 95% CI, three samples

The parser asserts the number of rows it finds against the table layout, so a silently dropped
or shifted row stops the run instead of producing a misaligned comparison.
"""
import os
import re

import pandas as pd

import cohort as C

JD = C.JD
TXT = f"{JD}/texts/medicina-62-01752.txt"
OUT = f"{JD}/results/paper"

OR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\)")

LEVELS = {
    "age_group": C.AGE_LABELS, "sex": ["Male", "Female"], "ethnicity": C.ETH_LEVELS,
    "area": ["Urban", "Rural"], "sector": ["Public", "Private"], "ari_type": C.TYPE_LEVELS,
    "year": [str(y) for y in C.YEARS]}
REF = {"total": {"age_group": "20-29", "sex": "Male", "ethnicity": "Mestizo", "area": "Urban",
                 "sector": "Public", "ari_type": C.TYPE_LEVELS[0], "year": "2015"}}
REF["adult"] = dict(REF["total"])
REF["pediatric"] = dict(REF["total"], age_group="0-4")
AGES = {"total": C.AGE_LABELS, "pediatric": C.AGE_LABELS[:4], "adult": C.AGE_LABELS[4:]}


def or_rows(sample):
    rows = []
    for v in C.PAPER_VARS:
        levels = AGES[sample] if v == "age_group" else LEVELS[v]
        for lv in levels:
            if lv != REF[sample][v]:
                rows.append((v, lv))
    return rows


def parse_or(text):
    out = []
    marks = [("total", "Table 5. Crude odds"), ("pediatric", "Table 6. Crude odds"),
             ("adult", "Table 7. Crude odds")]
    for i, (sample, mark) in enumerate(marks):
        s = text.index(mark)
        e = text.index("Note: OR, odds ratio", s)
        seg = text[s:e]
        m = OR_RE.findall(seg)
        rows = or_rows(sample)
        assert len(m) == 2 * len(rows), (sample, len(m), 2 * len(rows))
        for k, (v, lv) in enumerate(rows):
            o, a = m[2 * k], m[2 * k + 1]
            out.append(dict(sample=sample, variable=v, level=lv,
                            or_=float(o[0]), or_lo=float(o[1]), or_hi=float(o[2]),
                            aor=float(a[0]), aor_lo=float(a[1]), aor_hi=float(a[2])))
    return pd.DataFrame(out)


def parse_table1(text):
    """Table 1 rows carry 5 counts: pediatric alive/dead, adult alive/dead, total; age rows
    carry 3 (the sample they do not belong to is printed as N/A)."""
    s = text.index("Table 1. General characteristics")
    e = text.index("Note: IQR, interquartile range", s)
    seg = text[s:e]
    seg = re.sub(r"https://doi\.org/\S+|Medicina 2026, 62, 1752|\d+ of 25|Table 1\. Cont\.", " ", seg)
    lines = [ln.strip() for ln in seg.split("\n") if ln.strip()]
    num = re.compile(r"^\d{1,3}(?:,\d{3})*$|^\d+$")
    pct = re.compile(r"^\d+(?:\.\d+)?%$")
    order = ([("age_group", l) for l in C.AGE_LABELS] + [("sex", l) for l in LEVELS["sex"]]
             + [("ethnicity", l) for l in C.ETH_LEVELS] + [("area", l) for l in LEVELS["area"]]
             + [("sector", l) for l in LEVELS["sector"]]
             + [("ari_type", l) for l in C.TYPE_LEVELS] + [("year", l) for l in LEVELS["year"]])
    # walk the lines: a count is a number line immediately followed by a percentage line
    counts = []
    start = lines.index("Age groups")
    i = start
    while i < len(lines) - 1:
        if num.match(lines[i]) and pct.match(lines[i + 1]):
            counts.append(int(lines[i].replace(",", "")))
            i += 2
        else:
            i += 1
    rows, k = [], 0
    for v, lv in order:
        if v == "age_group":
            a, dth, tot = counts[k:k + 3]
            k += 3
            ped = lv in C.AGE_LABELS[:4]
            rows.append(dict(variable=v, level=lv,
                             ped_alive=a if ped else 0, ped_dead=dth if ped else 0,
                             adult_alive=0 if ped else a, adult_dead=0 if ped else dth,
                             total=tot))
        else:
            pa, pdd, aa, ad, tot = counts[k:k + 5]
            k += 5
            rows.append(dict(variable=v, level=lv, ped_alive=pa, ped_dead=pdd, adult_alive=aa,
                             adult_dead=ad, total=tot))
    assert k == len(counts), (k, len(counts))
    t = pd.DataFrame(rows)
    assert (t.ped_alive + t.ped_dead + t.adult_alive + t.adult_dead == t.total).all()
    for v, g in t.groupby("variable"):
        assert g.total.sum() == 576195, (v, g.total.sum())
    return t


def main():
    os.makedirs(OUT, exist_ok=True)
    text = open(TXT, errors="ignore").read()
    o = parse_or(text)
    o.to_csv(f"{OUT}/or_tables.csv", index=False)
    t = parse_table1(text)
    t.to_csv(f"{OUT}/table1.csv", index=False)
    print("OR rows:", o.groupby("sample").size().to_dict(), "| Table 1 rows:", len(t))


if __name__ == "__main__":
    main()
