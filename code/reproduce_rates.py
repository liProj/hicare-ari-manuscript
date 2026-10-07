"""Reproduce the epidemiological tables of Medicina 62:1752.

  Table 2  crude ARI hospitalisation / in-hospital mortality rates by province and year
  Table 3  the same by type of infection
  Table 4  age-adjusted rates (WHO world standard) for total / paediatric / adult populations,
           and the joinpoint annual percentage changes (APC)

Denominators: INEC population estimates and projections, 2024 revision (data/pop/).
Numerators are counted by admission year and by the province where the hospital is located --
the only combination that reproduces the printed rates.

The joinpoint part is a re-implementation (the article used NCI Joinpoint 6.1.0): log-linear
segmented regression on the nine annual rates, grid search over joinpoint years, zero or one
joinpoint, selection by BIC.  The printed APCs are recovered only when each annual log-rate is
weighted by its event count (Poisson variance) -- the unweighted fit is reported alongside to
show the difference.  The article's confidence limits come from Joinpoint's empirical-quantile
method and are not reproduced; ours are parametric.

Outputs: results/tables/rates_table2.csv, rates_table3.csv, rates_table4.csv,
         joinpoint.csv, rates_reproduction_summary.csv
"""
import os
import re

import numpy as np
import pandas as pd

import cohort as C

JD = C.JD
TAB = f"{JD}/results/tables"
POP = f"{JD}/data/pop"
YEARS = C.YEARS
PROV_ORDER = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13",
              "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]
# age groups for direct standardisation: the article's own ten groups.  Of the groupings tried
# (five-year groups open at 65/70/75/80/85/100, ten-year groups, and the Segi, US-2000 and
# European standards) this is the one closest to the printed rates.
AGE5 = list(C.AGE_LABELS)
AGE_EDGES = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 200]
NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def numbers_between(text, start, end):
    seg = text[text.index(start):text.index(end, text.index(start) + 10)]
    out = []
    for ln in seg.split("\n"):
        ln = ln.strip().replace("−", "-")
        if NUM.match(ln):
            v = float(ln)
            if not (2015 <= v <= 2023 and "." not in ln):
                out.append(v)
    return out


def parse_printed():
    t = open(f"{JD}/texts/medicina-62-01752.txt", errors="ignore").read()
    t2 = numbers_between(t, "Table 2. Acute respiratory infections hospitalization rates",
                         "Table 3. Acute respiratory infections hospitalization rates")
    assert len(t2) == 500, len(t2)
    t2 = np.array(t2).reshape(2, 25, 10)
    t3 = numbers_between(t, "Table 3. Acute respiratory infections hospitalization rates",
                         "3.3. Temporal Trends")
    assert len(t3) == 220, len(t3)
    t3 = np.array(t3).reshape(2, 11, 10)
    seg = t[t.index("Table 4. Age-adjusted"):t.index("Annual percentage change (APC) and average")]
    lines = [x.strip() for x in seg.split("\n") if x.strip()]
    t4 = {}
    for i, ln in enumerate(lines):
        if ln in [str(y) for y in YEARS] and i + 6 < len(lines) and all(
                NUM.match(lines[i + k]) for k in range(1, 7)):
            t4[int(ln)] = [float(lines[i + k]) for k in range(1, 7)]
    assert len(t4) == 9, t4
    return t2, t3, pd.DataFrame(t4, index=["hosp_total", "hosp_ped", "hosp_adult", "mort_total",
                                           "mort_ped", "mort_adult"]).T


# ------------------------------------------------------------------ joinpoint ---------------
def _fit_segments(x, ly, jp, w):
    """Weighted continuous piecewise-linear fit of log-rate with joinpoints `jp`."""
    cols = [np.ones_like(x), x] + [np.clip(x - j, 0, None) for j in jp]
    X = np.column_stack(cols)
    sw = np.sqrt(w / w.mean())
    Xw = X * sw[:, None]
    beta, *_ = np.linalg.lstsq(Xw, ly * sw, rcond=None)
    res = (ly - X @ beta) * sw
    return float(res @ res), beta, Xw


def joinpoint(years, rates, weights=None, max_jp=1, min_end=1):
    """`weights` = event counts: the variance of a log rate is ~1/count, which is the weighting
    Joinpoint applies when it is given counts and populations.  Unweighted if None."""
    x = np.asarray(years, float)
    ly = np.log(np.asarray(rates, float))
    n = len(x)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    fits = []
    sse0, b0, X0 = _fit_segments(x, ly, [], w)
    fits.append((0, [], sse0, b0, X0))
    if max_jp >= 1:
        best = None
        for j in x[min_end:n - min_end]:           # joinpoints at observed years
            sse, b, X = _fit_segments(x, ly, [j], w)
            if best is None or sse < best[2]:
                best = (1, [j], sse, b, X)
        fits.append(best)
    out = []
    for k, jp, sse, b, X in fits:
        p = 2 + 2 * k                                 # slopes+intercept and joinpoint location
        bic = np.log(sse / n) + p * np.log(n) / n     # Joinpoint's BIC
        bic3 = np.log(sse / n) + (2 + 3 * k) * np.log(n) / n
        dof = max(n - X.shape[1] - k, 1)
        cov = sse / dof * np.linalg.pinv(X.T @ X)
        segs, slope, var = [], b[1], cov[1, 1]
        bounds = [x[0]] + list(jp) + [x[-1]]
        for s in range(k + 1):
            if s > 0:
                slope = slope + b[1 + s]
                idx = list(range(1, 2 + s))
                var = cov[np.ix_(idx, idx)].sum()
            se = np.sqrt(max(var, 0))
            from scipy import stats
            tcrit = stats.t.ppf(0.975, dof)
            segs.append(dict(start=int(bounds[s]), end=int(bounds[s + 1]),
                             apc=100 * (np.exp(slope) - 1),
                             lo=100 * (np.exp(slope - tcrit * se) - 1),
                             hi=100 * (np.exp(slope + tcrit * se) - 1)))
        out.append(dict(k=k, bic=bic, bic3=bic3, sse=sse, segments=segs))
    return out


def main():
    os.makedirs(TAB, exist_ok=True)
    d = C.load()
    t2, t3, t4 = parse_printed()
    pp = pd.read_csv(f"{POP}/pop_province.csv", dtype={"prov_code": str})
    pn = pd.read_csv(f"{POP}/pop_national_age.csv")
    pn = pn[pn.sex == "both"]
    who = pd.read_csv(f"{POP}/who_standard.csv")
    nat = pn.groupby("year").population.sum()
    summ = []

    # ---- Table 2: province x year ---------------------------------------------------------
    rows = []
    for kind, mask in (("hospitalisation", np.ones(len(d), bool)), ("mortality", d.dead == 1)):
        cnt = (d[mask].groupby(["prov_ubi", "year"]).size().unstack("year")
               .reindex(index=PROV_ORDER, columns=YEARS).fillna(0))
        den = pp.pivot(index="prov_code", columns="year", values="population").reindex(PROV_ORDER)
        rate = cnt / den * 1e5
        tot = d[mask].groupby("year").size().reindex(YEARS) / nat.reindex(YEARS) * 1e5
        full = pd.concat([rate, tot.to_frame("Total").T])
        full["mean"] = full[YEARS].mean(axis=1)
        printed = t2[0 if kind == "hospitalisation" else 1]
        for i, key in enumerate(PROV_ORDER + ["Total"]):
            for j, col in enumerate(YEARS + ["mean"]):
                rows.append(dict(kind=kind, province=C.PROVINCES.get(key, "Total"), col=col,
                                 article=printed[i, j], ours=full.loc[key, col]))
    T2 = pd.DataFrame(rows)
    T2["exact"] = np.isclose(T2.ours.round(2), T2.article, atol=0.0051)
    T2.to_csv(f"{TAB}/rates_table2.csv", index=False)
    summ.append(dict(table="Table 2 (province × year crude rates)", numbers=len(T2),
                     exact_2dp=int(T2.exact.sum()),
                     max_abs_diff=float((T2.ours - T2.article).abs().max())))

    # ---- Table 3: type x year -------------------------------------------------------------
    rows = []
    for kind, mask in (("hospitalisation", np.ones(len(d), bool)), ("mortality", d.dead == 1)):
        cnt = (d[mask].groupby(["ari_type", "year"], observed=False).size().unstack("year")
               .reindex(index=C.TYPE_LEVELS, columns=YEARS).fillna(0))
        rate = cnt / nat.reindex(YEARS).values * 1e5
        tot = d[mask].groupby("year").size().reindex(YEARS) / nat.reindex(YEARS) * 1e5
        full = pd.concat([rate, tot.to_frame("Total").T])
        full["mean"] = full[YEARS].mean(axis=1)
        printed = t3[0 if kind == "hospitalisation" else 1]
        for i, key in enumerate(C.TYPE_LEVELS + ["Total"]):
            for j, col in enumerate(YEARS + ["mean"]):
                rows.append(dict(kind=kind, type=key, col=col, article=printed[i, j],
                                 ours=full.loc[key, col]))
    T3 = pd.DataFrame(rows)
    T3["exact"] = np.isclose(T3.ours.round(2), T3.article, atol=0.0051)
    T3.to_csv(f"{TAB}/rates_table3.csv", index=False)
    summ.append(dict(table="Table 3 (type × year crude rates)", numbers=len(T3),
                     exact_2dp=int(T3.exact.sum()),
                     max_abs_diff=float((T3.ours - T3.article).abs().max())))

    # ---- Table 4: age-adjusted rates -------------------------------------------------------
    def to5(g):
        lo = int(str(g).replace("+", "-").split("-")[0])
        return AGE5[int(np.searchsorted(AGE_EDGES, lo, side="right")) - 1]
    pn5 = pn.assign(a5=pn.age_group.map(to5)).groupby(["year", "a5"]).population.sum().unstack("a5")[AGE5]
    w = who.assign(a5=who.age_group.map(to5)).groupby("a5").weight_pct.sum().reindex(AGE5)
    age5 = d.age_group.astype(str)
    rows = []
    series, events = {}, {}
    for kind, mask in (("hosp", np.ones(len(d), bool)), ("mort", (d.dead == 1).values)):
        cnt = (pd.crosstab(d.year[mask], age5[mask]).reindex(index=YEARS, columns=AGE5)
               .fillna(0))
        spec = cnt / pn5.reindex(YEARS).values
        for pop, groups in (("total", AGE5), ("ped", AGE5[:4]), ("adult", AGE5[4:])):
            ww = w[groups] / w[groups].sum()
            adj = (spec[groups] * ww.values).sum(axis=1) * 1e5
            series[f"{kind}_{pop}"] = adj
            events[f"{kind}_{pop}"] = cnt[groups].sum(axis=1)
            for yv in YEARS:
                rows.append(dict(series=f"{kind}_{pop}", year=yv,
                                 article=t4.loc[yv, f"{kind}_{pop}"], ours=adj[yv]))
    T4 = pd.DataFrame(rows)
    T4["rel_diff"] = T4.ours / T4.article - 1
    T4["exact"] = np.isclose(T4.ours.round(2), T4.article, atol=0.0051)
    T4.to_csv(f"{TAB}/rates_table4.csv", index=False)
    summ.append(dict(table="Table 4 (age-adjusted rates)", numbers=len(T4),
                     exact_2dp=int(T4.exact.sum()),
                     max_abs_diff=float((T4.ours - T4.article).abs().max()),
                     max_rel_diff=float(T4.rel_diff.abs().max())))

    # ---- joinpoint -------------------------------------------------------------------------
    printed_apc = {   # series: [(start, end, APC)], as printed in Table 4
        "hosp_total": [(2015, 2021, 9.91), (2021, 2023, -12.35)],
        "hosp_ped": [(2015, 2021, -8.26), (2021, 2023, 53.39)],
        "hosp_adult": [(2015, 2021, 30.08), (2021, 2023, -53.09)],
        "mort_total": [(2015, 2021, 52.78), (2021, 2023, -71.28)],
        "mort_ped": [(2015, 2023, 1.20)],
        "mort_adult": [(2015, 2021, 55.10), (2021, 2023, -72.90)]}
    J = []
    for name, segs in printed_apc.items():
        for src, vals, wts in (
                ("article rates, unweighted", t4[name].values, None),
                ("article rates, count-weighted", t4[name].values, events[name].values),
                ("our rates, count-weighted", series[name].values, events[name].values)):
            fits = joinpoint(YEARS, vals, wts)
            k_art = len(segs) - 1
            chosen = min(fits, key=lambda f: f["bic"])
            same_k = [f for f in fits if f["k"] == k_art][0]
            for s_i, (a, b, apc) in enumerate(segs):
                sg = same_k["segments"][s_i]
                J.append(dict(series=name, rates=src, seg=f"{a}-{b}", article_apc=apc,
                              our_start=sg["start"], our_end=sg["end"], our_apc=sg["apc"],
                              our_lo=sg["lo"], our_hi=sg["hi"], k_article=k_art,
                              k_bic=chosen["k"]))
    J = pd.DataFrame(J)
    J.to_csv(f"{TAB}/joinpoint.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(f"{TAB}/rates_reproduction_summary.csv", index=False)
    print(S.to_string(index=False))
    print(T4.pivot(index="year", columns="series", values="ours").round(2).to_string())
    print(t4.to_string())
    print(J.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
