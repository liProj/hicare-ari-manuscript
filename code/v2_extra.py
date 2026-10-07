"""Additional computations behind the extended figure set of the second manuscript.

  traj      monthly trajectory of the state-space recalibration coefficients (forward chaining)
  shap      TreeSHAP of the LightGBM member with provider profiles
  geo       per-province discrimination and calibration on held-out provinces
  strainp   pandemic strain by province against the province's COVID-19 load
  reboundp  paediatric rebound by province (seasonal counterfactual per province)
  season    month-of-year profile of paediatric admissions, before and after the pandemic
  agemort   observed death proportion by age, infection group and period
  profile   provider profiles by facility class; profile vs ARI mortality per facility group

Outputs: results/v2/tables/x_*.csv
Usage: v2_extra.py [traj shap ...]
"""
import json
import os
import sys

os.environ.setdefault("ARI_THREADS", "8")

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import cohort as C
import evaluate as E
import features as F
import metrics as MT
import models as M
import v2_stack as S2

JD = F.JD
T = f"{JD}/results/v2/tables"
RS = np.random.RandomState(7)


def traj(f):
    cfg = json.load(open(f"{T}/dynamic_config.json"))
    y, ym = f.dead.astype(float), f.adm_ym.values
    extra = np.column_stack([(f.ari_type.astype(str) == "COVID-19").values.astype(float),
                             f.pediatric.values.astype(float)])
    P = E.load_dir(E.PRED, "forward", "total")
    V = E.load_dir(E.PVAL, "forward", "total")
    d, v = P[("lgbm", 0)], V[("lgbm", 0)]
    feats, q = cfg["feats"], cfg["q"]

    def phi(z, ex):
        cols = [np.ones_like(z)]
        if "s" in feats:
            cols.append(z)
        if "g" in feats:
            cols += [ex[:, 0], ex[:, 1]]
        return np.column_stack(cols)
    names = ["intercept"] + (["slope"] if "s" in feats else []) + (["covid", "paediatric"] if "g" in feats else [])
    rows = []
    for k in np.unique(d.fold.values):
        dk = d[d.fold == k]
        z, yy, mm, ex = S2.z_of(dk.p.values), y.loc[dk.index].values, ym[dk.index.values], extra[dk.index.values]
        dim = len(names)
        th = np.zeros(dim)
        if "s" in feats:
            th[1] = 1.0
        Pm = np.eye(dim) * 0.05
        wk = v[v.fold == k]
        wz, wy, wm, we = S2.z_of(wk.p.values), y.loc[wk.index].values, ym[wk.index.values], extra[wk.index.values]
        for t in np.unique(wm):
            mk = wm == t
            th, Pm = S2._upd(th, Pm, phi(wz[mk], we[mk]), wy[mk], wz[mk], q, feats)
        for t in np.unique(mm):
            mk = mm == t
            pred = expit(phi(z[mk], ex[mk]) @ th)
            r = dict(fold=int(k), ym=int(t), n=int(mk.sum()), deaths=int(yy[mk].sum()),
                     oe_static=float(yy[mk].sum() / expit(z[mk]).sum()), oe_dynamic=float(yy[mk].sum() / pred.sum()))
            sd = np.sqrt(np.diag(Pm + q * np.eye(dim)))
            for i, nm in enumerate(names):
                r[nm] = float(th[i])
                r[nm + "_sd"] = float(sd[i])
            rows.append(r)
            th, Pm = S2._upd(th, Pm, phi(z[mk], ex[mk]), yy[mk], z[mk], q, feats)
    R = pd.DataFrame(rows)
    R.to_csv(f"{T}/x_trajectory.csv", index=False)
    print(R.groupby("fold")[names + ["oe_static", "oe_dynamic"]].mean().round(3).to_string())


def shap(f):
    from sklearn.model_selection import StratifiedKFold
    cols = M.FULL + F.PROF
    skf = StratifiedKFold(5, shuffle=True, random_state=1000)
    tr, te = next(skf.split(np.arange(len(f)), f.dead.values))
    te = RS.choice(te, 60000, replace=False)
    M.fit_lgbm(f.iloc[tr], f.iloc[te], 0, cols)
    m = M.fit_lgbm.last
    _, Xte = M.cat_frames(f.iloc[tr], f.iloc[te], cols)
    sv = m.predict(Xte, pred_contrib=True, num_iteration=m.best_iteration)[:, :-1]
    grp = {c: g for g, cs in F.GROUPS.items() for c in cs}
    grp.update({c: "provider profile" for c in F.PROF})
    imp = pd.DataFrame(dict(feature=cols, group=[grp[c] for c in cols], mean_abs_shap=np.abs(sv).mean(0))).sort_values(
        "mean_abs_shap", ascending=False)
    imp.to_csv(f"{T}/x_shap_importance.csv", index=False)
    g = imp.groupby("group").mean_abs_shap.sum()
    (g / g.sum()).rename("share").reset_index().sort_values("share", ascending=False).to_csv(f"{T}/x_shap_groups.csv", index=False)
    sub = f.iloc[te]
    pd.DataFrame(dict(prof_mort=sub.prof_mort.values, shap_prof_mort=sv[:, cols.index("prof_mort")],
                      prof_los=sub.prof_los.values, shap_prof_los=sv[:, cols.index("prof_los")],
                      shap_profile=sv[:, [cols.index(c) for c in F.PROF]].sum(1),
                      shap_fac=sv[:, cols.index("fac")])).to_parquet(f"{T}/x_shap_profile.parquet", index=False)
    print(imp.head(14).to_string(index=False))
    print((g / g.sum()).sort_values(ascending=False).round(3).to_string())


def geo(f):
    y = f.dead.values.astype(float)
    rows = []
    for name in ("reference", "care_v1", "hicare"):
        d = pd.read_parquet(f"{JD}/results/v2/preds/geo/{name}.parquet")
        prov = f.prov_ubi.values[d.idx.values]
        yy, p = y[d.idx.values], d.p.values
        for pv in sorted(set(prov)):
            m = prov == pv
            if yy[m].sum() < 30:
                continue
            rows.append(dict(model=name, prov=C.PROVINCES[pv], n=int(m.sum()), deaths=int(yy[m].sum()),
                             auroc=MT.fast_auc(yy[m], p[m]), oe=float(yy[m].sum() / p[m].sum()),
                             logloss=MT.logloss(yy[m], p[m])))
    G = pd.DataFrame(rows)
    G.to_csv(f"{T}/x_geo_province.csv", index=False)
    w = G.pivot(index="prov", columns="model", values="auroc")
    print("AUROC wins vs reference: v1", int((w.care_v1 > w.reference).sum()), "hicare", int((w.hicare > w.reference).sum()),
          "| hicare > v1:", int((w.hicare > w.care_v1).sum()), "of", len(w))
    o = G.pivot(index="prov", columns="model", values="oe")
    print("median |log O/E|:", {c: round(float(np.abs(np.log(o[c])).median()), 3) for c in o})


def strainp(f):
    d = pd.read_parquet(f"{T}/strain_records.parquet")
    d["prov"] = d.fac.str[:2]
    pan = (d.ym >= 2020 * 12 + 2) & (d.ym <= 2021 * 12 + 11)
    pre = d.year <= 2019
    rows = []
    for pv, g in d.groupby("prov"):
        if pv not in C.PROVINCES:
            continue
        non = g[pan[g.index] & ~g.covid]
        O, Ex = non.y.sum(), non.e.sum()
        lo, hi = MT_byar(O, Ex)
        base = pre[g.index].sum() / 60.0 * 22           # ARI admissions expected in 22 months at the 2015-2019 rate
        rows.append(dict(prov=C.PROVINCES[pv], n=len(non), O=int(O), E=float(Ex), oe=float(O / Ex), lo=lo, hi=hi,
                         covid_adm=int((pan[g.index] & g.covid).sum()), load=float((pan[g.index] & g.covid).sum() / base)))
    R = pd.DataFrame(rows)
    R.to_csv(f"{T}/x_strain_province.csv", index=False)
    from scipy.stats import spearmanr
    rho, p = spearmanr(R.load, R.oe)
    json.dump(dict(rho=float(rho), p=float(p), n=len(R)), open(f"{T}/x_strain_province_corr.json", "w"))
    print(R.sort_values("oe", ascending=False).round(2).to_string(index=False))
    print("Spearman load vs O/E:", round(rho, 2), "p", round(p, 4))


def MT_byar(O, Ex):
    from scipy import stats
    lo = stats.chi2.ppf(0.025, 2 * O) / 2 / Ex if O > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * (O + 1)) / 2 / Ex
    return float(lo), float(hi)


def reboundp(f):
    import statsmodels.api as sm
    pp = pd.read_csv(f"{JD}/data/pop/pop_province.csv", dtype={"prov_code": str})
    ch = f[(f.pediatric == 1) & (f.ari_type.astype(str) != "COVID-19")]
    yms = np.arange(2015 * 12, 2024 * 12)
    yr, mo = yms // 12, yms % 12
    rows = []
    for pv, name in C.PROVINCES.items():
        cnt = ch[ch.prov_ubi == pv].groupby("adm_ym").size().reindex(yms, fill_value=0)
        pop = pp[pp.prov_code == pv].set_index("year").pop_0_19.reindex(yr).values.astype(float)
        if cnt[yr <= 2019].sum() < 300:
            continue
        X = np.column_stack([np.ones(len(yms))] + [(mo == k).astype(float) for k in range(1, 12)])
        tr = yr <= 2019
        g = sm.GLM(cnt.values[tr], X[tr], family=sm.families.Poisson(), offset=np.log(pop[tr])).fit()
        phi = max(1.0, g.pearson_chi2 / g.df_resid)
        B = RS.multivariate_normal(g.params, g.cov_params() * phi, 1500)
        mus = np.exp(B @ X.T + np.log(pop))
        sim = RS.negative_binomial(np.maximum(mus / (phi - 1), 1e-6), 1 / phi) if phi > 1.001 else RS.poisson(mus)
        mu = g.predict(X, offset=np.log(pop))
        for per, mk in (("Mar 2020–Dec 2021", (yms >= 2020 * 12 + 2) & (yr <= 2021)), ("2022", yr == 2022), ("2023", yr == 2023)):
            o, e, ss = cnt.values[mk].sum(), mu[mk].sum(), sim[:, mk].sum(1)
            rows.append(dict(prov=name, period=per, observed=int(o), expected=float(e), oe=float(o / e),
                             lo=float(o / np.quantile(ss, .975)), hi=float(o / max(np.quantile(ss, .025), 1e-9))))
    R = pd.DataFrame(rows)
    R.to_csv(f"{T}/x_rebound_province.csv", index=False)
    r23 = R[R.period == "2023"]
    print(r23.sort_values("oe", ascending=False).round(2).to_string(index=False))
    print("provinces above expectation in 2023 (PI excludes 1):", int((r23.lo > 1).sum()), "of", len(r23),
          "| below:", int((r23.hi < 1).sum()))


def season(f):
    t = f.ari_type.astype(str)
    ch = f[(f.pediatric == 1) & (t != "COVID-19")]
    rows = []
    for nm, g in (("All non-COVID ARI, 0–19 y", ch), ("Pneumonia, 1–9 y", ch[ch.ari_type.astype(str).str.contains("pneumonia") & (ch.age_cont >= 1) & (ch.age_cont < 10)]),
                  ("Bronchiolitis, <1 y", ch[(ch.ari_type.astype(str) == "Acute bronchiolitis") & (ch.age_cont < 1)])):
        # mes_ingr is stored as a string; convert before grouping so that months sort 1..12
        c = g.assign(mes_ingr=g.mes_ingr.astype(float).astype(int)).groupby(["year", "mes_ingr"]).size().unstack(fill_value=0)
        sh = c.div(c.sum(axis=1), axis=0)
        for yv in sh.index:
            for m in sh.columns:
                rows.append(dict(series=nm, year=int(yv), month=int(float(m)), share=float(sh.loc[yv, m]), n=int(c.loc[yv, m])))
    pd.DataFrame(rows).to_csv(f"{T}/x_seasonality.csv", index=False)
    print(pd.DataFrame(rows).groupby(["series", "year"]).n.sum().unstack().to_string())


def agemort(f):
    t = f.ari_type.astype(str)
    grp = np.where(t == "COVID-19", "COVID-19", np.where(t.str.contains("pneumonia"), "Pneumonia (non-COVID)", "Other ARI"))
    per = np.where(f.year <= 2019, "2015–2019", np.where(f.year <= 2021, "2020–2021", "2022–2023"))
    band = pd.cut(f.age_cont.values, [-1, 0.999, 4.999, 9.999, 19.999, 39.999, 49.999, 59.999, 69.999, 79.999, 200],
                  labels=["<1", "1–4", "5–9", "10–19", "20–39", "40–49", "50–59", "60–69", "70–79", "≥80"])
    d = pd.DataFrame(dict(grp=grp, per=per, band=band, y=f.dead.values, early=f.death_lt48h.values))
    A = d.groupby(["grp", "per", "band"], observed=True).agg(n=("y", "size"), deaths=("y", "sum"), early=("early", "sum")).reset_index()
    A.to_csv(f"{T}/x_age_mortality.csv", index=False)
    print(A[A.grp == "Pneumonia (non-COVID)"].pivot(index="band", columns="per", values="deaths").to_string())


def profile(f):
    from v2_medical import CLASE
    cl = pd.to_numeric(f.clase, errors="coerce")
    feats = ["prof_n", "prof_mort", "prof_mort48", "prof_los", "prof_ped", "prof_old", "prof_outcanton",
             "prof_ch_obstetric", "prof_ch_injury", "prof_ch_neoplasm", "prof_ch_circulatory", "prof_ch_infect"]
    g = f.assign(clase_i=cl).groupby("clase_i")
    A = g[feats].mean()
    A["n"] = g.size()
    A["ari_death_pct"] = g.dead.mean() * 100
    A = A[A.n >= 900]
    A.index = [CLASE.get(int(i), str(i)) for i in A.index]
    A.to_csv(f"{T}/x_profile_by_class.csv")
    fg = f.groupby("fac").agg(n=("dead", "size"), deaths=("dead", "sum"), prof_mort=("prof_mort", "mean"),
                              prof_los=("prof_los", "mean"), clase=("clase", "first"), sector=("sector", "first")).reset_index()
    fg = fg[(fg.n >= 300) & fg.prof_mort.notna()]
    fg["ari_death_pct"] = fg.deaths / fg.n * 100
    fg.to_csv(f"{T}/x_profile_facility.csv", index=False)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(fg.prof_mort, fg.ari_death_pct)
    json.dump(dict(rho=float(rho), n=len(fg)), open(f"{T}/x_profile_corr.json", "w"))
    print(A.round(3).to_string())
    print("facility groups:", len(fg), "Spearman(all-cause non-ARI mortality, ARI mortality) =", round(rho, 2))


def main():
    f = F.build()
    todo = dict(traj=traj, shap=shap, geo=geo, strainp=strainp, reboundp=reboundp, season=season, agemort=agemort,
                profile=profile)
    for k in ([a for a in sys.argv[1:]] or list(todo)):
        print(f"\n===== {k} =====", flush=True)
        todo[k](f)


if __name__ == "__main__":
    main()
