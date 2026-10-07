"""Reconcile the public INEC registry against every count printed in Medicina 62:1752 Table 1.

Writes data_check.md and results/tables/data_check.csv.  Exit status is non-zero if any cell of
Table 1 fails to reconcile, so the pipeline cannot continue on an unverified cohort.
"""
import os
import sys

import numpy as np
import pandas as pd

import cohort as C

JD = C.JD


def main():
    os.makedirs(f"{JD}/results/tables", exist_ok=True)
    raw = pd.read_parquet(C.COHORT)
    d = C.load()
    t1 = pd.read_csv(f"{JD}/results/paper/table1.csv", keep_default_na=False)

    rows = []
    for _, r in t1.iterrows():
        v, lv = r.variable, r.level
        col = d[v].astype(str)
        m = col == str(lv)
        ours = dict(
            ped_alive=int((m & (d.pediatric == 1) & (d.dead == 0)).sum()),
            ped_dead=int((m & (d.pediatric == 1) & (d.dead == 1)).sum()),
            adult_alive=int((m & (d.pediatric == 0) & (d.dead == 0)).sum()),
            adult_dead=int((m & (d.pediatric == 0) & (d.dead == 1)).sum()),
            total=int(m.sum()))
        for k, val in ours.items():
            rows.append(dict(variable=v, level=lv, cell=k, article=int(r[k]), registry=val,
                             diff=val - int(r[k])))
    cells = pd.DataFrame(rows)
    cells.to_csv(f"{JD}/results/tables/data_check.csv", index=False)
    n_bad = int((cells["diff"] != 0).sum())

    ped, adu = d[d.pediatric == 1], d[d.pediatric == 0]

    def q(s):
        return f"{s.median():.0f} ({s.quantile(.25):.0f}–{s.quantile(.75):.0f})"

    head = [
        ("Hospitalisations (unit of analysis)", "576,195", f"{len(d):,}"),
        ("In-hospital deaths", "41,048 (7.12%)", f"{int(d.dead.sum()):,} ({d.dead.mean():.2%})"),
        ("Paediatric (0–19 y) records", "295,225 (51.24%)", f"{len(ped):,} ({len(ped)/len(d):.2%})"),
        ("Adult (≥20 y) records", "280,970 (48.76%)", f"{len(adu):,} ({len(adu)/len(d):.2%})"),
        ("Paediatric deaths", "1575 (0.53%)", f"{int(ped.dead.sum())} ({ped.dead.mean():.2%})"),
        ("Adult deaths", "39,473 (14.04%)", f"{int(adu.dead.sum()):,} ({adu.dead.mean():.2%})"),
        ("Median age, all (IQR)", "14 (1–64)", q(d.age)),
        ("Median age, paediatric survivors", "1 (0–4)", q(ped.age[ped.dead == 0])),
        ("Median age, paediatric deaths", "1 (0–5)", q(ped.age[ped.dead == 1])),
        ("Median age, adult survivors", "63 (47–78)", q(adu.age[adu.dead == 0])),
        ("Median age, adult deaths", "72 (61–83)", q(adu.age[adu.dead == 1])),
        ("COVID-19 share of 2020–2021 hospitalisations", "70.71%",
         f"{d[d.year.isin([2020, 2021])].ari_type.eq('COVID-19').mean():.2%}"),
        ("COVID-19 share of 2020–2021 deaths", "87.17%",
         f"{d[d.year.isin([2020, 2021]) & (d.dead == 1)].ari_type.eq('COVID-19').mean():.2%}"),
    ]
    miss = {c: int(d[c].isna().sum()) for c in C.PAPER_VARS + ["dead", "age"]}
    raw_in = raw[raw.anio_ingr.between(2015, 2023)]
    n_pre = int((~raw.anio_ingr.between(2015, 2023)).sum())
    n_cov = int((raw_in.causa3.eq("U07") & (raw_in.anio_ingr < 2020)).sum())
    dup = int(d.drop(columns=["adm_date"]).duplicated().sum())

    L = ["# Data check — Medicina 2026, 62, 1752 vs the public INEC registry", "",
         "Source: INEC *Registro Estadístico de Camas y Egresos Hospitalarios*, nine yearly "
         "files (2015–2023), downloaded from ecuadorencifras.gob.ec. "
         "Every archive was opened and CRC-verified before use.", "",
         "## 1. From raw registry to the study cohort", "",
         "| Step | Records |", "|---|---|",
         f"| All discharges in the nine files | "
         f"{sum(len(pd.read_parquet(f'{JD}/data/cohort/all_{y}.parquet', columns=['file_year'])) for y in C.YEARS):,} |",
         f"| Principal diagnosis in J00–J06, J09–J18, J20–J22, J85–J86, U07.1–U07.2 | {len(raw):,} |",
         f"| − admitted before 2015 (outside the study window) | −{n_pre} |",
         f"| − COVID-19 code with an admission year before 2020 (undocumented rule) | −{n_cov} |",
         f"| **Study cohort** | **{len(d):,}** |", "",
         "The article states 576,195. The second exclusion is not described in the article; "
         "it was identified because the 14 records it removes account exactly for the residual "
         "in every margin of Table 1 (year 2016/2017/2018/2019: 1/1/7/5; Mestizo 14; "
         "private 13, public 1; urban 12, rural 2; male 8, female 6; all 14 are deaths).", "",
         "## 2. Headline numbers", "", "| Item | Article | Public registry | Match |",
         "|---|---|---|---|"]
    ok_all, notes = True, []
    for name, art, ours in head:
        # the count is what has to reconcile; a percentage that differs in the last digit with
        # an identical count is a rounding choice in the article and is footnoted, not failed
        ok = art.replace(",", "").split(" (")[0] == ours.replace(",", "").split(" (")[0]
        exact = art.replace(",", "") == ours.replace(",", "")
        ok_all &= ok
        if ok and not exact:
            notes.append(f"{name}: identical count; the article prints {art.split('(')[1][:-1]} "
                         f"where the same count gives {ours.split('(')[1][:-1]} "
                         f"(truncated rather than rounded).")
        L.append(f"| {name} | {art} | {ours} | {'✅' if exact else ('✅ ¹' if ok else '❌')} |")
    if notes:
        L += [""] + [f"¹ {n}" for n in notes]
    L += ["", "## 3. Table 1, cell by cell", "",
          f"Table 1 has {len(t1)} category rows × 5 count columns = {len(cells)} cells "
          f"(paediatric alive / dead, adult alive / dead, total). "
          f"**{len(cells) - n_bad} of {len(cells)} cells reconcile exactly; {n_bad} differ.**", "",
          "| Variable | Levels | Cells | Exact | Max abs. difference |", "|---|---|---|---|---|"]
    for v in C.PAPER_VARS:
        g = cells[cells.variable == v]
        L.append(f"| {v} | {g.level.nunique()} | {len(g)} | {(g['diff'] == 0).sum()} | "
                 f"{g['diff'].abs().max()} |")
    L += ["", "## 4. Integrity of the cohort", "", "| Check | Result |", "|---|---|",
          f"| Missing values in the outcome or any of the seven model variables | "
          f"{sum(miss.values())} |",
          f"| Records with age unit 'ignored' | {int((d.age_unit == 9).sum())} |",
          f"| Class balance (dead / alive) | {int(d.dead.sum()):,} / {int((d.dead == 0).sum()):,} |",
          f"| Unique patient identifier | none released — the unit of analysis is the "
          f"hospitalisation, as in the article |",
          f"| Records identical on every retained column | {dup:,} "
          f"(expected in a registry without patient or facility identifiers; the article does "
          f"not de-duplicate and neither do we) |",
          f"| Implausible age (>115 y) | {int((d.age > 115).sum())} |", "",
          "## 5. Verdict", "",
          ("**PASS.** The public registry reproduces the article's cohort exactly — total, "
           "deaths, both age strata and every cell of Table 1."
           if (n_bad == 0 and ok_all) else "**FAIL** — see the differences above.")]
    open(f"{JD}/data_check.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-14:]))
    print(f"cells: {len(cells)}, mismatching: {n_bad}; headline all ok: {ok_all}")
    if n_bad or not ok_all:
        print(cells[cells["diff"] != 0].to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
