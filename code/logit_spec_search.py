"""Which adjusted model did the article fit?

The crude odds ratios of Tables 5-7 follow from the Table 1 counts and reproduce; the adjusted
ones are close but not identical under the seven-variable model the article describes.  This
script refits the total-sample model under a list of candidate departures from the stated
specification and scores each against the 35 printed adjusted ORs (and their 70 CI limits).

Score = number of printed aORs matched at two decimals, and the RMS log-ratio.
Output: results/tables/logit_spec_search.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import cohort as C
from reproduce_logit import design

JD = C.JD


def dummies(s, drop_first=True):
    return pd.get_dummies(s.astype(str), drop_first=drop_first).astype(float)


def main():
    sample = sys.argv[1] if len(sys.argv) > 1 else "total"
    d = C.sample(C.load(), sample)
    paper = pd.read_csv(f"{JD}/results/paper/or_tables.csv", keep_default_na=False)
    p = paper[paper["sample"] == sample].set_index(["variable", "level"])
    base = design(d, sample, C.PAPER_VARS)
    y = d.dead.values.astype(float)
    los = d.dia_estad.fillna(0).clip(lower=0)
    cands = {
        "stated (7 variables)": None,
        "+ length of stay (days)": los.to_frame("los"),
        "+ log(1+length of stay)": np.log1p(los).to_frame("llos"),
        "+ hospital province": dummies(d.prov_ubi),
        "+ province of residence": dummies(d.prov_res),
        "+ hospital area (urban/rural)": dummies(d.area_ubi),
        "+ nationality": dummies(d.nac_pac),
        "+ admission month": dummies(d.mes_ingr),
        "+ facility class": dummies(d.clase),
        "+ managing entity": dummies(d.entidad),
        "+ sector in 3 levels (extra: non-profit)": (d.sector3 == 3).astype(float).to_frame("np"),
        "+ continuous age": d.age.to_frame("age"),
        "+ death timing excluded (<48 h deaths dropped)": "drop48",
        "year = registry file year": "fileyear",
    }
    rows = []
    for name, extra in cands.items():
        X, yy = base, y
        if isinstance(extra, str) and extra == "drop48":
            keep = (d.death_lt48h == 0).values
            X, yy = base[keep], y[keep]
        elif isinstance(extra, str) and extra == "fileyear":
            dd = d.copy()
            dd["year"] = dd.file_year
            X = design(dd, sample, C.PAPER_VARS)
        elif extra is not None:
            X = pd.concat([base, extra.set_axis([("x", c) for c in extra.columns], axis=1)],
                          axis=1)
        try:
            m = sm.Logit(yy, X.values.astype(float)).fit(disp=0, maxiter=200)
        except Exception as e:
            print(name, "failed", repr(e)[:80], flush=True)
            continue
        ci = m.conf_int()
        est, lo, hi = {}, {}, {}
        for k, col in enumerate(X.columns):
            if col in p.index:
                est[col], lo[col], hi[col] = np.exp(m.params[k]), np.exp(ci[k, 0]), np.exp(ci[k, 1])
        e = pd.Series(est).reindex(p.index)
        l_ = pd.Series(lo).reindex(p.index)
        h = pd.Series(hi).reindex(p.index)
        allr = np.concatenate([e.values, l_.values, h.values])
        allp = np.concatenate([p.aor.values, p.aor_lo.values, p.aor_hi.values])
        rows.append(dict(spec=name, n=len(yy),
                         aor_exact_2dp=int(np.isclose(np.round(e.values, 2), p.aor.values,
                                                      atol=0.0051).sum()),
                         all_exact_2dp=int(np.isclose(np.round(allr, 2), allp, atol=0.0051).sum()),
                         n_aor=len(p), n_all=len(allp),
                         rms_log_ratio=float(np.sqrt(np.mean(np.log(e.values / p.aor.values) ** 2))),
                         covid_aor=float(e[("ari_type", "COVID-19")]),
                         private_aor=float(e[("sector", "Private")])))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows).sort_values("rms_log_ratio")
    os.makedirs(f"{JD}/results/tables", exist_ok=True)
    out.to_csv(f"{JD}/results/tables/logit_spec_search_{sample}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
