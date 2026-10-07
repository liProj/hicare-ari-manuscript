"""Reproduce Tables 5-7 of Medicina 62:1752: crude and adjusted odds ratios for in-hospital
death in the total, paediatric and adult samples.

The article fitted binary logistic regression in SPSS 25 with seven categorical predictors.
The same models are fitted here by maximum likelihood (statsmodels Logit) and every printed
OR, aOR and confidence limit is compared with the re-estimated one.

Outputs: results/tables/logit_reproduction.csv  (one row per coefficient per sample)
         results/tables/logit_reproduction_summary.csv
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

import cohort as C
from parse_paper_tables import AGES, LEVELS, REF

JD = C.JD


def design(d, sample, variables):
    cols = {}
    for v in variables:
        levels = AGES[sample] if v == "age_group" else LEVELS[v]
        s = d[v].astype(str)
        for lv in levels:
            if lv != REF[sample][v]:
                cols[(v, lv)] = (s == lv).astype(np.float64).values
    X = pd.DataFrame(cols, index=d.index)
    X.columns = pd.MultiIndex.from_tuples(X.columns)
    X.insert(0, ("const", ""), 1.0)
    return X


def fit(d, sample, variables):
    X = design(d, sample, variables)
    m = sm.Logit(d.dead.values.astype(float), X.values).fit(disp=0, maxiter=200)
    ci = m.conf_int()
    out = []
    for k, (v, lv) in enumerate(X.columns):
        if v == "const":
            continue
        out.append(dict(variable=v, level=lv, est=np.exp(m.params[k]), lo=np.exp(ci[k, 0]),
                        hi=np.exp(ci[k, 1]), p=m.pvalues[k]))
    return pd.DataFrame(out), m


def main():
    os.makedirs(f"{JD}/results/tables", exist_ok=True)
    d = C.load()
    paper = pd.read_csv(f"{JD}/results/paper/or_tables.csv", keep_default_na=False)
    rows, summ = [], []
    for sample in ("total", "pediatric", "adult"):
        ds = C.sample(d, sample)
        adj, m = fit(ds, sample, C.PAPER_VARS)
        crude = pd.concat([fit(ds, sample, [v])[0] for v in C.PAPER_VARS], ignore_index=True)
        p = paper[paper["sample"] == sample]
        mg = (p.merge(crude.rename(columns=dict(est="r_or", lo="r_or_lo", hi="r_or_hi", p="r_p")),
                      on=["variable", "level"], validate="1:1")
               .merge(adj.rename(columns=dict(est="r_aor", lo="r_aor_lo", hi="r_aor_hi",
                                              p="r_ap")),
                      on=["variable", "level"], validate="1:1"))
        assert len(mg) == len(p)
        rows.append(mg)
        pairs = [("or_", "r_or"), ("or_lo", "r_or_lo"), ("or_hi", "r_or_hi"),
                 ("aor", "r_aor"), ("aor_lo", "r_aor_lo"), ("aor_hi", "r_aor_hi")]
        for kind, sel in (("crude", pairs[:3]), ("adjusted", pairs[3:])):
            a = np.concatenate([mg[x].values for x, _ in sel])
            b = np.concatenate([mg[y].values for _, y in sel])
            exact = np.isclose(np.round(b, 2), a, atol=0.0051)
            summ.append(dict(sample=sample, n=len(ds), events=int(ds.dead.sum()), kind=kind,
                             numbers=len(a), exact_2dp=int(exact.sum()),
                             within_0p02=int((np.abs(np.round(b, 2) - a) <= 0.0201).sum()),
                             max_abs_log_ratio=float(np.max(np.abs(np.log(b / a)))),
                             median_abs_rel_diff=float(np.median(np.abs(b / a - 1)))))
        print(sample, "n", len(ds), "pseudo-R2", round(m.prsquared, 4), "llf", round(m.llf, 1))
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(f"{JD}/results/tables/logit_reproduction.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(f"{JD}/results/tables/logit_reproduction_summary.csv", index=False)
    print(S.to_string(index=False))
    bad = R[~np.isclose(np.round(R.r_aor, 2), R.aor, atol=0.0051)]
    print("\nadjusted ORs that differ at 2 dp:")
    print(bad[["sample", "variable", "level", "aor", "r_aor", "aor_lo", "r_aor_lo", "aor_hi",
               "r_aor_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
