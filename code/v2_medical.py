"""Population-health analyses built on case-mix standardisation.

  E1  risk-standardised mortality: provinces, provider types and facility groups
  E2  pandemic strain: excess case-mix-adjusted fatality among NON-COVID ARI admissions
  E3  inequity decomposition: crude gap -> clinical mix -> place of care
  E4  post-pandemic paediatric rebound against a pre-pandemic seasonal counterfactual
  E5  risk phenotypes: a shallow tree described on held-out records

Expected deaths come from out-of-fold predictions of LightGBM case-mix models
(`lgbm@clin`: age, sex, diagnosis, season, national epidemic context, year;
 `lgbm@clin_prov`: the same plus everything known about the provider).  Group variables whose
gaps are being studied (ethnicity, area of residence, referral) are in neither model.

Outputs: results/v2/tables/*.csv
Usage: v2_medical.py [E1 E2 ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit, gammaln, logit

import cohort as C
import features as F
import metrics as MT
import models as M

JD = F.JD
OUT = f"{JD}/results/v2/tables"
PRED = f"{JD}/results/preds/cv/total"
RS = np.random.RandomState(20261006)
ENTIDAD = {1: "Ministry of Public Health", 2: "Ministry of Justice", 3: "Armed forces",
           4: "Ministry of Education", 5: "Other ministries", 6: "Social security (IESS)",
           7: "Social-security annexes", 8: "Peasant social insurance", 9: "Other public",
           10: "Provincial councils", 11: "Municipalities", 12: "Universities",
           13: "Junta de Beneficencia", 14: "Red Cross", 15: "SOLCA (cancer society)",
           16: "Fiscomisional", 17: "Private non-profit", 18: "Private for-profit"}
CLASE = {1: "Basic hospital", 2: "General hospital", 3: "Infectious-disease hospital",
         4: "Obstetric hospital", 5: "Paediatric hospital", 6: "Psychiatric hospital",
         7: "Dermatology hospital", 8: "Oncology hospital", 9: "Pulmonology hospital",
         10: "Geriatric hospital", 11: "Specialties hospital", 12: "General clinic",
         13: "Obstetric clinic", 14: "Paediatric clinic", 15: "Trauma clinic",
         16: "Psychiatric clinic", 17: "Other specialised clinic", 32: "Day facility"}


def expected(f, variant):
    """Out-of-fold case-mix predictions, recalibrated so that expected = observed overall
    (indirect standardisation) with one logistic map on the out-of-fold predictions."""
    d = pd.read_parquet(f"{PRED}/lgbm@{variant}_s0.parquet").set_index("idx").sort_index()
    assert len(d) == len(f)
    z = logit(MT.clip(d.p.values))
    a, b = MT.platt(z, f.dead.values.astype(float))
    return expit(a + b * z)


def coarse_expected(f):
    """Case-mix as the reference article could adjust it: age band, sex, type of infection and
    year (5-fold out-of-fold logistic regression)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    X = pd.get_dummies(pd.DataFrame(dict(a=f.age_group.astype(str), s=f.sex.astype(str),
                                         t=f.ari_type.astype(str), y=f.year.astype(str))),
                       drop_first=True).values.astype(np.float32)
    y = f.dead.values
    out = np.zeros(len(f))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000).split(X, y):
        m = LogisticRegression(penalty=None, max_iter=2000).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


# ------------------------------------------------------------------ empirical Bayes SMR -----
def eb_gamma(O, E):
    """O_j ~ Poisson(E_j * theta_j), theta_j ~ Gamma(a, a).  Returns a (prior precision)."""
    O, E = np.asarray(O, float), np.asarray(E, float)

    def nll(la):
        a = np.exp(la)
        return -np.sum(gammaln(O + a) - gammaln(a) - gammaln(O + 1) + a * np.log(a / (a + E))
                       + O * np.log(E / (a + E)))
    r = optimize.minimize_scalar(nll, bounds=(-3, 9), method="bounded")
    return float(np.exp(r.x))


def smr_table(f, key, E, min_n=300, min_e=5, label=None):
    g = pd.DataFrame(dict(k=f[key].values, O=f.dead.values, E=E)).groupby("k").agg(
        n=("O", "size"), O=("O", "sum"), E=("E", "sum")).reset_index()
    g = g[(g.n >= min_n) & (g.E >= min_e)].copy()
    a = eb_gamma(g.O, g.E)
    g["smr"] = g.O / g.E
    g["smr_eb"] = (g.O + a) / (g.E + a)
    g["lo"] = stats.gamma.ppf(0.025, g.O + a, scale=1 / (g.E + a))
    g["hi"] = stats.gamma.ppf(0.975, g.O + a, scale=1 / (g.E + a))
    g["flag"] = np.where(g.lo > 1, "high", np.where(g.hi < 1, "low", "as expected"))
    g["prior_a"] = a
    if label is not None:
        g["name"] = g.k.map(label)
    return g.sort_values("smr_eb", ascending=False)


def E1(f):
    e_rich, e_coarse = expected(f, "clin"), coarse_expected(f)
    f = f.assign(prov_name=f.prov_ubi.map(C.PROVINCES),
                 entidad_i=pd.to_numeric(f.entidad, errors="coerce"),
                 clase_i=pd.to_numeric(f.clase, errors="coerce"))
    summ = []
    for key, lab, mn in (("prov_ubi", C.PROVINCES, 300), ("entidad_i", ENTIDAD, 500),
                         ("clase_i", CLASE, 500), ("fac", None, 300)):
        for nm, E in (("crude", np.full(len(f), f.dead.mean())), ("coarse", e_coarse),
                      ("rich", e_rich)):
            t = smr_table(f, key, E, min_n=mn, label=lab)
            t.to_csv(f"{OUT}/smr_{key}_{nm}.csv", index=False)
            summ.append(dict(unit=key, casemix=nm, units=len(t), prior_a=t.prior_a.iloc[0],
                             between_sd=float(np.sqrt(1 / t.prior_a.iloc[0])),
                             high=int((t.flag == "high").sum()), low=int((t.flag == "low").sum()),
                             smr_p10=float(t.smr_eb.quantile(.1)), smr_p90=float(t.smr_eb.quantile(.9)),
                             excess_in_high=float((t.O - t.E)[t.flag == "high"].sum()),
                             excess_all_above=float(np.clip(t.O - t.E, 0, None).sum()),
                             deaths=float(t.O.sum())))
    S = pd.DataFrame(summ)
    S.to_csv(f"{OUT}/smr_summary.csv", index=False)
    # reclassification of facility groups: coarse vs rich case-mix
    a = pd.read_csv(f"{OUT}/smr_fac_coarse.csv").set_index("k")
    b = pd.read_csv(f"{OUT}/smr_fac_rich.csv").set_index("k")
    j = a.join(b, lsuffix="_coarse", rsuffix="_rich", how="inner")
    pd.crosstab(j.flag_coarse, j.flag_rich).to_csv(f"{OUT}/smr_fac_reclassification.csv")
    # province SMR by period and by age stratum (rich case-mix)
    rows = []
    per = np.where(f.year <= 2019, "2015-2019", np.where(f.year <= 2021, "2020-2021", "2022-2023"))
    for nm, mask in ([(p, per == p) for p in ("2015-2019", "2020-2021", "2022-2023")]
                     + [("paediatric", f.pediatric.values == 1), ("adult", f.pediatric.values == 0)]):
        t = smr_table(f[mask], "prov_ubi", e_rich[mask], min_n=100, min_e=2, label=C.PROVINCES)
        rows.append(t.assign(stratum=nm))
    pd.concat(rows).to_csv(f"{OUT}/smr_province_strata.csv", index=False)
    print(S.round(3).to_string(index=False))
    print(pd.crosstab(j.flag_coarse, j.flag_rich).to_string())
    print(pd.read_csv(f"{OUT}/smr_prov_ubi_rich.csv")[["name", "n", "O", "E", "smr_eb", "lo", "hi", "flag"]]
          .round(3).to_string(index=False))


# ------------------------------------------------------------------ E2 pandemic strain -------
def byar(O, E):
    O = np.asarray(O, float)
    lo = O * (1 - 1 / (9 * np.maximum(O, 1e-9)) - 1.96 / (3 * np.sqrt(np.maximum(O, 1e-9)))) ** 3
    hi = (O + 1) * (1 - 1 / (9 * (O + 1)) + 1.96 / (3 * np.sqrt(O + 1))) ** 3
    return np.where(O > 0, lo, 0) / E, hi / E


def E2(f):
    from sklearn.model_selection import StratifiedKFold
    cols = F.CLIN
    pre = (f.year <= 2019).values
    post = ~pre
    pos = np.arange(len(f))
    p = np.zeros(len(f))
    ppre = pos[pre]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000).split(ppre, f.dead.values[ppre]):
        p[ppre[te]] = M.fit_lgbm(f.iloc[ppre[tr]], f.iloc[ppre[te]], 11, cols)
    p[post] = M.fit_lgbm(f.iloc[ppre], f.iloc[pos[post]], 12, cols)
    a, b = MT.platt(logit(MT.clip(p[pre])), f.dead.values[pre].astype(float))
    p = expit(a + b * logit(MT.clip(p)))
    t = f.ari_type.astype(str)
    d = pd.DataFrame(dict(ym=f.adm_ym.values, y=f.dead.values, e=p, covid=(t == "COVID-19").values,
                          ped=f.pediatric.values, pneu=t.str.contains("pneumonia").values,
                          fac=f.fac.values, year=f.year.values))
    d.to_parquet(f"{OUT}/strain_records.parquet", index=False)
    non = d[~d.covid]
    rows = []
    for nm, m in (("all non-COVID", np.ones(len(non), bool)), ("adult non-COVID", (non.ped == 0).values),
                  ("paediatric non-COVID", (non.ped == 1).values),
                  ("non-COVID pneumonia", non.pneu.values), ("non-COVID non-pneumonia", (~non.pneu).values)):
        g = non[m].groupby("ym").agg(n=("y", "size"), O=("y", "sum"), E=("e", "sum")).reset_index()
        lo, hi = byar(g.O, g.E)
        rows.append(g.assign(group=nm, oe=g.O / g.E, lo=lo, hi=hi))
    Mo = pd.concat(rows)
    Mo.to_csv(f"{OUT}/strain_monthly.csv", index=False)
    per = np.where(non.year <= 2019, "2015-2019", np.where(non.year <= 2021, "2020-2021", "2022-2023"))
    rows = []
    for nm, m in (("all non-COVID", np.ones(len(non), bool)), ("adult non-COVID", (non.ped == 0).values),
                  ("paediatric non-COVID", (non.ped == 1).values),
                  ("non-COVID pneumonia", non.pneu.values), ("non-COVID non-pneumonia", (~non.pneu).values)):
        for pr in ("2015-2019", "2020-2021", "2022-2023"):
            s = non[m & (per == pr)]
            lo, hi = byar([s.y.sum()], s.e.sum())
            rows.append(dict(group=nm, period=pr, n=len(s), O=int(s.y.sum()), E=float(s.e.sum()),
                             oe=float(s.y.sum() / s.e.sum()), lo=float(lo[0]), hi=float(hi[0]),
                             excess=float(s.y.sum() - s.e.sum())))
    Pp = pd.DataFrame(rows)
    Pp.to_csv(f"{OUT}/strain_period.csv", index=False)
    # dose-response: COVID load of the facility group in the admission month
    base = d[d.year <= 2019].groupby("fac").size() / 60.0             # mean monthly ARI pre-pandemic
    cov = d[d.covid].groupby(["fac", "ym"]).size().rename("ncov")
    pan = non[(non.ym >= 2020 * 12 + 2) & (non.ym <= 2021 * 12 + 11)].copy()
    pan = pan.join(cov, on=["fac", "ym"]).join(base.rename("base"), on="fac")
    pan["load"] = pan.ncov.fillna(0) / pan.base.where(pan.base >= 2)
    pan = pan.dropna(subset=["load"])
    bins = [-0.01, 0, 0.5, 1, 2, 4, np.inf]
    labs = ["none", "0–0.5×", "0.5–1×", "1–2×", "2–4×", ">4×"]
    pan["load_cat"] = pd.cut(pan.load, bins, labels=labs)
    rows = []
    for strat, m in (("all", np.ones(len(pan), bool)), ("adult", (pan.ped == 0).values),
                     ("paediatric", (pan.ped == 1).values),
                     # sensitivity: diagnoses that an untested COVID-19 case is unlikely to be
                     # coded as (upper respiratory infection, bronchitis, bronchiolitis, influenza)
                     ("non-pneumonia", (~pan.pneu).values)):
        g = pan[m].groupby("load_cat", observed=True).agg(n=("y", "size"), O=("y", "sum"), E=("e", "sum")).reset_index()
        lo, hi = byar(g.O, g.E)
        rows.append(g.assign(stratum=strat, oe=g.O / g.E, lo=lo, hi=hi))
    Ld = pd.concat(rows)
    Ld.to_csv(f"{OUT}/strain_load.csv", index=False)
    print(Pp.round(3).to_string(index=False))
    print(Ld.round(3).to_string(index=False))


# ------------------------------------------------------------------ E3 inequity --------------
def E3(f):
    e1, e2 = expected(f, "clin"), expected(f, "clin_prov")
    y = f.dead.values.astype(float)
    early = (f.death_lt48h.values == 1)
    contrasts = [
        ("Rural vs urban residence", f.area.astype(str) == "Rural", f.area.astype(str) == "Urban"),
        ("Indigenous vs Mestizo", f.ethnicity.astype(str) == "Indigenous", f.ethnicity.astype(str) == "Mestizo"),
        ("Afro-Ecuadorian vs Mestizo", f.ethnicity.astype(str) == "Black/Afro", f.ethnicity.astype(str) == "Mestizo"),
        ("Montubio vs Mestizo", f.ethnicity.astype(str) == "Montubio", f.ethnicity.astype(str) == "Mestizo"),
        ("Treated outside own canton vs within", f.same_cant.astype(str) == "0", f.same_cant.astype(str) == "1"),
        ("Public vs private sector", f.sector.astype(str) == "Public", f.sector.astype(str) == "Private"),
    ]
    rows = []
    for strat, sm in (("Paediatric", f.pediatric.values == 1), ("Adult", f.pediatric.values == 0)):
        for name, g, r in contrasts:
            ig, ir = np.flatnonzero(g.values & sm), np.flatnonzero(r.values & sm)

            def rr(ig_, ir_):
                og, orr = y[ig_].sum(), y[ir_].sum()
                return ((og / len(ig_)) / (orr / len(ir_)),
                        (og / e1[ig_].sum()) / (orr / e1[ir_].sum()),
                        (og / e2[ig_].sum()) / (orr / e2[ir_].sum()),
                        (early[ig_].sum() / max(og, 1)) / (early[ir_].sum() / max(orr, 1)))
            pt = rr(ig, ir)
            bs = np.array([rr(RS.choice(ig, len(ig)), RS.choice(ir, len(ir))) for _ in range(400)])
            lo, hi = np.nanquantile(bs, .025, axis=0), np.nanquantile(bs, .975, axis=0)
            expl_clin = 1 - np.log(pt[1]) / np.log(pt[0]) if abs(np.log(pt[0])) > 0.02 else np.nan
            expl_prov = (np.log(pt[1]) - np.log(pt[2])) / np.log(pt[0]) if abs(np.log(pt[0])) > 0.02 else np.nan
            rows.append(dict(stratum=strat, contrast=name, n_group=len(ig), deaths_group=int(y[ig].sum()),
                             n_ref=len(ir), deaths_ref=int(y[ir].sum()),
                             rr_crude=pt[0], rr_crude_lo=lo[0], rr_crude_hi=hi[0],
                             rr_clin=pt[1], rr_clin_lo=lo[1], rr_clin_hi=hi[1],
                             rr_clin_prov=pt[2], rr_clin_prov_lo=lo[2], rr_clin_prov_hi=hi[2],
                             share_gap_clinical=expl_clin, share_gap_provider=expl_prov,
                             early_share_group=early[ig].sum() / max(y[ig].sum(), 1),
                             early_share_ref=early[ir].sum() / max(y[ir].sum(), 1),
                             early_ratio=pt[3], early_ratio_lo=lo[3], early_ratio_hi=hi[3]))
    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT}/inequity.csv", index=False)
    print(T[["stratum", "contrast", "deaths_group", "rr_crude", "rr_clin", "rr_clin_prov", "early_share_group",
             "early_share_ref"]].round(3).to_string(index=False))


# ------------------------------------------------------------------ E4 paediatric rebound ----
def E4(f):
    import statsmodels.api as sm
    pop = pd.read_csv(f"{JD}/data/pop/pop_national_age.csv")
    pop = pop[pop.sex == "both"]
    lo_age = pop.age_group.map(lambda g: int(str(g).replace("+", "-").split("-")[0]))
    popy = {"<1 y": pop[lo_age == 0].groupby("year").population.sum(),
            "1–4 y": pop[lo_age == 0].groupby("year").population.sum(),
            "5–9 y": pop[lo_age == 5].groupby("year").population.sum(),
            "10–19 y": pop[lo_age.isin([10, 15])].groupby("year").population.sum(),
            "≥20 y": pop[lo_age >= 20].groupby("year").population.sum()}
    t = f.ari_type.astype(str)
    synd = np.where(t == "COVID-19", "COVID-19", np.where(t == "Acute bronchiolitis", "Bronchiolitis",
            np.where(t.str.contains("pneumonia"), "Pneumonia", np.where(t == "Influenza", "Influenza",
            np.where(t.str.contains("upper"), "Upper respiratory", "Other lower respiratory")))))
    band = pd.cut(f.age_cont.values, [-1, 0.999, 4.999, 9.999, 19.999, 200],
                  labels=["<1 y", "1–4 y", "5–9 y", "10–19 y", "≥20 y"]).astype(str)
    d = pd.DataFrame(dict(ym=f.adm_ym.values, band=band, synd=synd, y=f.dead.values))
    d = d[d.synd != "COVID-19"]
    yms = np.arange(2015 * 12, 2024 * 12)
    rows, summ = [], []
    series = [(b, s) for b in ["<1 y", "1–4 y", "5–9 y", "10–19 y", "≥20 y"] for s in ["All non-COVID ARI"]]
    series += [(b, s) for b in ["<1 y", "1–4 y"] for s in ["Bronchiolitis", "Pneumonia", "Influenza", "Upper respiratory"]]
    series += [("0–19 y", "All non-COVID ARI"), ("0–19 y deaths", "All non-COVID ARI")]
    for b, s in series:
        if b == "0–19 y":
            m = d.band != "≥20 y"
            pp = sum(popy[k] for k in ["1–4 y", "5–9 y", "10–19 y"])
            cnt = d[m].groupby("ym").size()
        elif b == "0–19 y deaths":
            m = (d.band != "≥20 y") & (d.y == 1)
            pp = sum(popy[k] for k in ["1–4 y", "5–9 y", "10–19 y"])
            cnt = d[m].groupby("ym").size()
        else:
            m = (d.band == b) & ((d.synd == s) if s != "All non-COVID ARI" else True)
            pp = popy[b]
            cnt = d[m].groupby("ym").size()
        cnt = cnt.reindex(yms, fill_value=0)
        yr, mo = yms // 12, yms % 12
        season = [(mo == k).astype(float) for k in range(1, 12)]
        off = np.log(pp.reindex(yr).values.astype(float))
        tr = yr <= 2019
        # primary counterfactual: pre-pandemic seasonal rate, no trend -- a log-linear trend
        # fitted on five years and extrapolated four years ahead is fragile (it predicts an
        # eight-fold rise of coded influenza); the trend model is kept as a sensitivity analysis
        for mname, X in (("seasonal", np.column_stack([np.ones(len(yms))] + season)),
                         ("seasonal+trend", np.column_stack([np.ones(len(yms)), (yms - yms[0]) / 12.0] + season))):
            g = sm.GLM(cnt.values[tr], X[tr], family=sm.families.Poisson(), offset=off[tr]).fit()
            phi = max(1.0, g.pearson_chi2 / g.df_resid)
            mu = g.predict(X, offset=off)
            # prediction interval by simulation: parameter uncertainty + negative-binomial noise
            B = RS.multivariate_normal(g.params, g.cov_params() * phi, 2000)
            mus = np.exp(B @ X.T + off)
            if phi > 1.001:
                sim = RS.negative_binomial(np.maximum(mus / (phi - 1), 1e-6), 1 / phi)
            else:
                sim = RS.poisson(mus)
            lo, hi = np.quantile(sim, .025, axis=0), np.quantile(sim, .975, axis=0)
            rows.append(pd.DataFrame(dict(model=mname, band=b, syndrome=s, ym=yms, year=yr, month=mo + 1,
                                          observed=cnt.values, expected=mu, lo=lo, hi=hi)))
            for nm, mk in (("2015-2019", yr <= 2019), ("Mar 2020–Dec 2021", (yms >= 2020 * 12 + 2) & (yr <= 2021)),
                           ("2022", yr == 2022), ("2023", yr == 2023), ("Mar 2020–Dec 2023", yms >= 2020 * 12 + 2)):
                o, e = cnt.values[mk].sum(), mu[mk].sum()
                ss = sim[:, mk].sum(1)
                summ.append(dict(model=mname, band=b, syndrome=s, period=nm, observed=int(o), expected=float(e),
                                 oe=float(o / e), oe_lo=float(o / np.quantile(ss, .975)),
                                 oe_hi=float(o / max(np.quantile(ss, .025), 1e-9)), excess=float(o - e),
                                 dispersion=float(phi)))
    pd.concat(rows).to_csv(f"{OUT}/rebound_monthly.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(f"{OUT}/rebound_summary.csv", index=False)
    # age shift of bronchiolitis and pneumonia admissions in young children
    ag = pd.DataFrame(dict(year=f.year.values, age=f.age_cont.values, synd=synd))
    ag = ag[(ag.age < 5) & ag.synd.isin(["Bronchiolitis", "Pneumonia"])]
    A = ag.groupby(["synd", "year"]).age.agg(["mean", "median", "size"]).reset_index()
    A["share_1plus"] = ag.assign(o=(ag.age >= 1)).groupby(["synd", "year"]).o.mean().values
    A.to_csv(f"{OUT}/rebound_age_shift.csv", index=False)
    P1 = S[S.model == "seasonal"].drop(columns="model")
    print(P1[P1.syndrome == "All non-COVID ARI"].round(3).to_string(index=False))
    print(P1[(P1.syndrome != "All non-COVID ARI") & P1.period.isin(["2022", "2023"])].round(3).to_string(index=False))
    print(A.round(3).to_string(index=False))


# ------------------------------------------------------------------ E5 phenotypes ------------
def E5(f):
    from sklearn.tree import DecisionTreeClassifier
    t = f.ari_type.astype(str)
    clase = pd.to_numeric(f.clase, errors="coerce")
    X = pd.DataFrame({
        "age (years)": f.age_cont.values,
        "COVID-19": (t == "COVID-19").astype(int).values,
        "pneumonia (any)": t.str.contains("pneumonia").astype(int).values,
        "upper respiratory / bronchitis": t.isin(["Acute upper respiratory infections", "Acute bronchitis"]).astype(int).values,
        "public sector": (f.sector.astype(str) == "Public").astype(int).values,
        "basic hospital or general clinic": clase.isin([1, 12]).astype(int).values,
        "admitted 2020–2021": f.year.isin([2020, 2021]).astype(int).values,
        "male": (f.sex.astype(str) == "Male").astype(int).values})
    y = f.dead.values
    idx = RS.permutation(len(f))
    tr, te = idx[: int(0.8 * len(f))], idx[int(0.8 * len(f)):]
    tree = DecisionTreeClassifier(max_leaf_nodes=14, min_samples_leaf=3000, random_state=0).fit(X.iloc[tr], y[tr])
    leaf = tree.apply(X.iloc[te])
    tt = tree.tree_

    def path(node_id):
        # walk from the root to the leaf, collecting the splits
        conds, node = [], 0
        while tt.children_left[node] != -1:
            feat, thr = X.columns[tt.feature[node]], tt.threshold[node]
            left = _contains(tt, tt.children_left[node], node_id)
            if feat == "age (years)":
                conds.append((feat, "<=" if left else ">", thr))
            else:
                conds.append((feat, "no" if left else "yes", None))
            node = tt.children_left[node] if left else tt.children_right[node]
        return conds

    def describe(conds):
        lo, hi, flags = 0.0, 200.0, {}
        for feat, op, thr in conds:
            if feat == "age (years)":
                if op == "<=":
                    hi = min(hi, thr)
                else:
                    lo = max(lo, thr)
            else:
                flags[feat] = op
        age = (f"age {lo:.0f}–{hi:.0f} y" if lo > 0 and hi < 200 else f"age ≤{hi:.0f} y" if hi < 200
               else f"age >{lo:.0f} y" if lo > 0 else "any age")
        if 0 < hi < 1.5 and lo == 0:
            age = f"age <{max(hi, 1):.0f} y"
        txt = [age] + [(k if v == "yes" else f"not {k}") for k, v in flags.items()]
        return "; ".join(txt)

    rows = []
    for lf in np.unique(leaf):
        m = leaf == lf
        rows.append(dict(phenotype=describe(path(lf)), n=int(m.sum()), deaths=int(y[te][m].sum()),
                         death_pct=100 * y[te][m].mean(), share_admissions=m.mean(),
                         share_deaths=y[te][m].sum() / y[te].sum()))
    T = pd.DataFrame(rows).sort_values("death_pct", ascending=False)
    T["cum_share_admissions"] = T.share_admissions.cumsum()
    T["cum_share_deaths"] = T.share_deaths.cumsum()
    T.to_csv(f"{OUT}/phenotypes.csv", index=False)
    print(T.round(3).to_string(index=False))


def _contains(tt, node, target):
    if node == target:
        return True
    if tt.children_left[node] == -1:
        return False
    return _contains(tt, tt.children_left[node], target) or _contains(tt, tt.children_right[node], target)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = F.build()
    want = [a for a in sys.argv[1:]] or ["E1", "E2", "E3", "E4", "E5"]
    for k in want:
        print(f"\n===== {k} =====", flush=True)
        {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5}[k](f)


if __name__ == "__main__":
    main()
