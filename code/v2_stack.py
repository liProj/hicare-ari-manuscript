"""HiCARE calibration and temporal-maintenance analyses.

Level 1: LightGBM and XGBoost for the main HiCARE configuration; EW also uses TabM.
Level 2a: Logistic stacking with child and COVID-19 interactions.
Level 2b: Shallow residual boosting on specified variables, initialised from stacked logits.
Level 2c: Unpenalised logistic projection on corrected scores and marginal indicators.
           Projection uses all nested rows, including residual early-stopping records.
           A finite, converged fit matches marginal totals in its fitting data, not
           necessarily in held-out populations or every intersecting subgroup.
Level 3: Monthly state-space updating for temporal evaluation.

Nested predictions isolate direct base-model fitting from outer test records. Historical
context is precomputed and does not fully isolate earlier held-out outcomes. The geographic
protocol uses separate profile-ensemble configurations without the full second layer.
Legacy internal names are retained to match archived aggregate result files.

Outputs: results/v2/preds/<protocol>/<name>.parquet (idx, fold, p),
         results/v2/tables/v2_metrics.csv, v2_subgroups.csv, v2_winloss.csv, v2_boot.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import cohort as C
import evaluate as E
import features as F
import metrics as MT

JD = F.JD
NEST = f"{JD}/results/v2/nested"
OUTP = f"{JD}/results/v2/preds"
TAB = f"{JD}/results/v2/tables"
MEMBERS = ["lgbm", "xgb", "tabm"]
# members of HiCARE's second layer; fixed on the command line (--members lgbm,xgb) so that the
# same composition is used under every validation design
SECOND = list(MEMBERS)
GROUP_COLS = ["age_cont", "sexo", "etnia", "area_res", "sector3", "ari_type_c", "entidad", "clase",
              "prov_ubi", "mes_ingr"]
SUBGROUPS = [("age_group", C.AGE_LABELS), ("sex", ["Male", "Female"]), ("ethnicity", C.ETH_LEVELS),
             ("area", ["Urban", "Rural"]), ("sector", ["Public", "Private"]),
             ("ari_type", C.TYPE_LEVELS), ("year", [str(y) for y in C.YEARS])]


def z_of(p):
    return logit(MT.clip(np.asarray(p, float)))


def design_a(Z, ped, covid):
    return np.column_stack([Z, Z * ped[:, None], Z * covid[:, None], ped, covid])


def fit_stack(Z, y, ped, covid):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1000.0, max_iter=2000).fit(design_a(Z, ped, covid), y)


def group_frame(f, idx, za, with_year):
    X = pd.DataFrame({"za": za})
    for c in GROUP_COLS:
        v = f[c].values[idx]
        X[c] = v.astype(np.float32) if c == "age_cont" else pd.Categorical(v.astype(str))
    if with_year:
        X["year"] = pd.Categorical(f.year.values[idx].astype(str))
    return X


def fit_mcboost(Xtr, ytr, init, seed, hold_frac=0.15):
    """Boosting from the stacked log-odds; the number of rounds is chosen on a hold-out part of
    the nested rows.  Returns (model, best_iteration, hold-out positions)."""
    import lightgbm as lgb
    rs = np.random.RandomState(seed)
    hold = rs.rand(len(ytr)) < hold_frac
    p = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=1000,
             lambda_l2=20.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             cat_smooth=50, cat_l2=20, max_cat_to_onehot=8, num_threads=8, seed=seed, verbose=-1)
    dtr = lgb.Dataset(Xtr[~hold], ytr[~hold], init_score=init[~hold])
    dva = lgb.Dataset(Xtr[hold], ytr[hold], init_score=init[hold], reference=dtr)
    m = lgb.train(p, dtr, 500, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m, m.best_iteration, np.flatnonzero(hold)


def proj_design(f, idx, zb, with_year):
    """Design of the multiaccuracy projection: the boosted log-odds plus an indicator for every
    level of the reference article's own categorical variables."""
    cols = [zb]
    for col, levels in SUBGROUPS:
        if col == "year" and not with_year:
            continue
        v = f[col].astype(str).values[idx]
        for lv in levels[1:]:
            cols.append((v == lv).astype(np.float64))
    return np.column_stack(cols)


def fit_projection(X, y):
    """Unpenalised maximum-likelihood logistic fit.  Its score equations force the fitted
    probabilities to reproduce the observed number of deaths within every subgroup level --
    the property that makes the reference model 'perfectly' calibrated in its own subgroups."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000, tol=1e-8).fit(X, y)


def align_cats(Xtr, Xte):
    for c in Xtr.columns:
        if Xtr[c].dtype.name == "category":
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
    return Xte


def second_layer(f, y, test, nested, members, with_year, seed=0):
    """test: {member: frame(idx-indexed: fold, p)}; nested: {member: frame(outer, idx, p)}.
    Returns dict name -> frame(fold, p) and, per fold, hold-out predictions of the final layer."""
    idx_te = test[members[0]].index
    fold = test[members[0]].fold.values
    ped_all = f.pediatric.values.astype(float)
    cov_all = (f.ari_type.astype(str) == "COVID-19").values.astype(float)
    out = {k: np.zeros(len(idx_te)) for k in ("stack", "mcboost", "hicare")}
    holds, info = [], []
    for k in np.unique(fold):
        te = fold == k
        ite = idx_te.values[te]
        nk = [nested[m][nested[m].outer == k].set_index("idx").sort_index() for m in members]
        itr = nk[0].index.values
        assert all(n.index.equals(nk[0].index) for n in nk)
        assert len(np.intersect1d(itr, ite)) == 0, "nested rows overlap the test fold"
        Ztr = np.column_stack([z_of(n.p.values) for n in nk])
        Zte = np.column_stack([z_of(test[m].p.values[te]) for m in members])
        ytr = y.values[itr]
        A = fit_stack(Ztr, ytr, ped_all[itr], cov_all[itr])
        za_tr = A.decision_function(design_a(Ztr, ped_all[itr], cov_all[itr]))
        za_te = A.decision_function(design_a(Zte, ped_all[ite], cov_all[ite]))
        out["stack"][te] = expit(za_te)
        Xtr = group_frame(f, itr, za_tr, with_year)
        Xte = align_cats(Xtr, group_frame(f, ite, za_te, with_year))
        mb, it, hold = fit_mcboost(Xtr, ytr, za_tr, seed * 10 + int(k) + 20)
        zb_te = za_te + mb.predict(Xte, num_iteration=it, raw_score=True)
        out["mcboost"][te] = expit(zb_te)
        # level 2c: multiaccuracy projection, fitted on ALL nested rows.  (Fitting it on the
        # 15% hold-out alone was tried first: ~40 parameters on ~5,000 deaths give an intercept
        # whose sampling error, ~0.015, is as large as the bias it is meant to remove.)
        zb_tr = za_tr + mb.predict(Xtr, num_iteration=it, raw_score=True)
        Pj = fit_projection(proj_design(f, itr, zb_tr, with_year), ytr)
        out["hicare"][te] = Pj.predict_proba(proj_design(f, ite, zb_te, with_year))[:, 1]
        out["mcboost"][te] = expit(zb_te)
        ph = Pj.predict_proba(proj_design(f, itr[hold], zb_tr[hold], with_year))[:, 1]
        holds.append(pd.DataFrame(dict(fold=k, p=ph), index=itr[hold]))
        info.append(dict(fold=int(k), rounds=int(it), stack_coef=A.coef_[0].round(3).tolist()))
    frames = {n: pd.DataFrame(dict(fold=fold, p=v), index=idx_te) for n, v in out.items()}
    return frames, pd.concat(holds), info


# ------------------------------------------------------------------ dynamic recalibration ----
def dynamic(d, y, ym, extra, warm=None, q=0.01, feats="isg", p0=0.05):
    """State-space logistic recalibration.  logit p_t = phi' theta_t, theta_t = theta_{t-1} + N(0, qI).
    Month t is predicted with the state filtered through month t-1, then the state is updated
    with month t's outcomes (Laplace approximation).  `feats`: "i" intercept, "is" intercept and
    slope, "isg" plus offsets for COVID-19 and for children."""
    def phi(z, ex):
        cols = [np.ones_like(z)]
        if "s" in feats:
            cols.append(z)
        if "g" in feats:
            cols += [ex[:, 0], ex[:, 1]]
        return np.column_stack(cols)
    out = d.copy()
    newp = d.p.values.copy()
    z_all, y_all, m_all = z_of(d.p.values), y.loc[d.index].values, ym[d.index.values]
    ex_all = extra[d.index.values]
    for k in np.unique(d.fold.values):
        fk = d.fold.values == k
        zs, ys, ms, es = z_all[fk], y_all[fk], m_all[fk], ex_all[fk]
        pos = np.flatnonzero(fk)
        dim = phi(zs[:1], es[:1]).shape[1]
        theta = np.zeros(dim)
        if "s" in feats:
            theta[1] = 1.0
        P = np.eye(dim) * p0

        def update(th, P_, Phi, yy):
            prior_prec = np.linalg.inv(P_ + q * np.eye(dim))
            t_new = th.copy()
            for _ in range(25):
                mu = expit(Phi @ t_new if "s" in feats else Phi @ t_new + 0)
                g = Phi.T @ (yy - mu) - prior_prec @ (t_new - th)
                H = Phi.T @ (Phi * (mu * (1 - mu))[:, None]) + prior_prec
                step = np.linalg.solve(H, g)
                t_new = t_new + step
                if np.max(np.abs(step)) < 1e-8:
                    break
            return t_new, np.linalg.inv(H)

        def lin(Phi, th, zz):
            # without a slope term the log-odds enter as an offset
            return Phi @ th if "s" in feats else Phi @ th + zz

        if warm is not None:
            wk = warm[warm.fold == k]
            wz, wy, wm = z_of(wk.p.values), y.loc[wk.index].values, ym[wk.index.values]
            we = extra[wk.index.values]
            for t in np.unique(wm):
                mk = wm == t
                Phi = phi(wz[mk], we[mk])
                theta, P = _upd(theta, P, Phi, wy[mk], wz[mk], q, feats)
        for t in np.unique(ms):
            mk = ms == t
            Phi = phi(zs[mk], es[mk])
            newp[pos[mk]] = expit(lin(Phi, theta, zs[mk]))
            theta, P = _upd(theta, P, Phi, ys[mk], zs[mk], q, feats)
    out["p"] = newp
    return out


def _upd(theta, P, Phi, yy, zz, q, feats):
    dim = len(theta)
    prior_prec = np.linalg.inv(P + q * np.eye(dim))
    off = 0.0 if "s" in feats else zz
    t_new = theta.copy()
    H = prior_prec
    for _ in range(25):
        mu = expit(Phi @ t_new + off)
        g = Phi.T @ (yy - mu) - prior_prec @ (t_new - theta)
        H = Phi.T @ (Phi * (mu * (1 - mu))[:, None]) + prior_prec
        step = np.linalg.solve(H, g)
        t_new = t_new + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return t_new, np.linalg.inv(H)


def select_dynamic(f, y, ym, extra):
    """Choose (features, drift variance) on forward-chaining test years 2018-2021 of the
    first-version LightGBM member -- never on 2022-2023."""
    P = E.load_dir(E.PRED, "forward", "total")
    V = E.load_dir(E.PVAL, "forward", "total")
    d, v = P[("lgbm", 0)], V[("lgbm", 0)]
    dev = d[d.fold <= 2021]
    rows = []
    for feats in ("i", "is", "isg"):
        for q in (0.001, 0.01, 0.1):
            r = dynamic(dev, y, ym, extra, warm=v, q=q, feats=feats)
            mt = MT.all_metrics(y.loc[r.index].values, r.p.values, "total")
            rows.append(dict(feats=feats, q=q, logloss=mt["logloss"], brier=mt["brier"],
                             cal_int=mt["cal_int"], cal_slope=mt["cal_slope"], ici=mt["ici"]))
    r = E.rolling(dev, y, ym, warm=v)
    mt = MT.all_metrics(y.loc[r.index].values, r.p.values, "total")
    rows.append(dict(feats="rolling window (v1)", q=np.nan, logloss=mt["logloss"], brier=mt["brier"],
                     cal_int=mt["cal_int"], cal_slope=mt["cal_slope"], ici=mt["ici"]))
    R = pd.DataFrame(rows)
    R.to_csv(f"{TAB}/dynamic_selection.csv", index=False)
    best = R[R.q.notna()].sort_values("logloss").iloc[0]
    cfg = dict(feats=best.feats, q=float(best.q))
    json.dump(cfg, open(f"{TAB}/dynamic_config.json", "w"))
    print(R.round(5).to_string(index=False), "\nselected:", cfg, flush=True)
    return cfg


# ------------------------------------------------------------------ metrics -------------------
def subgroup_metrics(f, y, d):
    """Calibration inside the 42 subgroups defined by the reference article's own variables."""
    sub = f.iloc[d.index.values]
    yy, p = y.loc[d.index].values, d.p.values
    rows = []
    for col, levels in SUBGROUPS:
        v = sub[col].astype(str).values
        for lv in levels:
            m = v == lv
            if yy[m].sum() < 25 or (1 - yy[m]).sum() < 25:
                continue
            rows.append(dict(variable=col, level=lv, n=int(m.sum()), deaths=int(yy[m].sum()),
                             oe=float(yy[m].mean() / p[m].mean()), ece=MT.ece(yy[m], p[m], 10),
                             auroc=MT.fast_auc(yy[m], p[m])))
    return pd.DataFrame(rows)


def summarise_sub(S):
    lo = np.abs(np.log(S.oe))
    return dict(sub_max_abs_log_oe=float(lo.max()), sub_mean_abs_log_oe=float(np.average(lo, weights=S.n)),
                sub_within_5pct=float((lo < np.log(1.05)).mean()),
                sub_max_ece=float(S[S.n >= 5000].ece.max()), sub_mean_ece=float(np.average(S.ece, weights=S.n)))


def save(d, protocol, name):
    os.makedirs(f"{OUTP}/{protocol}", exist_ok=True)
    x = d.copy()
    x.attrs = {}
    x.index.name = "idx"
    x.reset_index().to_parquet(f"{OUTP}/{protocol}/{name}.parquet", index=False)


def main():
    global SECOND
    for i, a in enumerate(sys.argv):
        if a == "--members":
            SECOND = sys.argv[i + 1].split(",")
    os.makedirs(TAB, exist_ok=True)
    f = F.build()
    y = f.dead.astype(float)
    ym = f.adm_ym.values
    extra = np.column_stack([(f.ari_type.astype(str) == "COVID-19").values.astype(float),
                             f.pediatric.values.astype(float)])
    strata = {"total": f.index, "pediatric": f.index[f.pediatric == 1], "adult": f.index[f.pediatric == 0]}
    long, subs, notes = [], [], {}

    def score(d, protocol, name):
        for s, ids in strata.items():
            dd = d.loc[d.index.intersection(ids)]
            r = MT.all_metrics(y.loc[dd.index].values, dd.p.values, s)
            long.extend(dict(protocol=protocol, sample=s, model=name, metric=k, value=v) for k, v in r.items())
        if protocol != "forward":
            S = subgroup_metrics(f, y, d)
            subs.append(S.assign(protocol=protocol, model=name))
            for k, v in summarise_sub(S).items():
                long.append(dict(protocol=protocol, sample="total", model=name, metric=k, value=v))
        save(d, protocol, name)

    # ---------------- cross-validation ------------------------------------------------------
    Pt, Vt = E.load_dir(E.PRED, "cv", "total"), E.load_dir(E.PVAL, "cv", "total")
    members = [m for m in SECOND if (f"{m}@prof", 0) in Pt and os.path.exists(f"{NEST}/cv/{m}@prof.parquet")]
    notes["cv_members"] = members
    print("cv members with nested predictions:", members, flush=True)
    score(Pt[("paper_lr", 0)], "cv", "reference")
    v1 = E.ensemble(Pt, Vt, y, MEMBERS, 0)
    if v1 is not None:
        score(v1[0], "cv", "care_v1")
    if all((f"{m}@prof", 0) in Pt for m in MEMBERS):
        score(E.ensemble(Pt, Vt, y, [f"{m}@prof" for m in MEMBERS], 0)[0], "cv", "equal_weight_prof")
    for m in MEMBERS:
        if (f"{m}@prof", 0) in Pt:
            score(Pt[(f"{m}@prof", 0)], "cv", f"{m}_prof")
    if members:
        test = {m: Pt[(f"{m}@prof", 0)] for m in members}
        nested = {m: pd.read_parquet(f"{NEST}/cv/{m}@prof.parquet") for m in members}
        fr, _, info = second_layer(f, y, test, nested, members, with_year=True)
        notes["cv_layer"] = info
        score(fr["stack"], "cv", "stack")
        score(fr["mcboost"], "cv", "stack+mcboost")
        score(fr["hicare"], "cv", "hicare")
        if len(members) > 1:                       # ablation: second layer on LightGBM alone
            fr1, _, _ = second_layer(f, y, {"lgbm": test["lgbm"]}, {"lgbm": nested["lgbm"]}, ["lgbm"], True)
            score(fr1["hicare"], "cv", "hicare_lgbm_only")
        fr2, _, _ = second_layer(f, y, test, nested, members, with_year=False)
        score(fr2["hicare"], "cv", "hicare_no_year")

    # ---------------- temporal ---------------------------------------------------------------
    Pt, Vt = E.load_dir(E.PRED, "temporal", "total"), E.load_dir(E.PVAL, "temporal", "total")
    cfg = select_dynamic(f, y, ym, extra)
    notes["dynamic"] = cfg
    ref_cands = {}
    for m in ("paper_lr", "paper_lr_carry", "paper_lr_noyear"):
        if (m, 0) in Pt:
            ref_cands[m] = Pt[(m, 0)]
            ref_cands[m + "+roll"] = E.rolling(Pt[(m, 0)], y, ym, warm=Vt.get((m, 0)))
            ref_cands[m + "+dyn"] = dynamic(Pt[(m, 0)], y, ym, extra, warm=Vt.get((m, 0)), **cfg)
    ll = {k: MT.logloss(y.loc[d.index].values, d.p.values) for k, d in ref_cands.items()}
    best = min(ll, key=ll.get)
    notes["temporal_best_reference"] = best
    notes["temporal_reference_logloss"] = {k: round(v, 5) for k, v in ll.items()}
    score(ref_cands[best], "temporal", "reference")
    for k, d in ref_cands.items():
        score(d, "temporal", f"ref:{k}")
    v1 = E.ensemble(Pt, Vt, y, MEMBERS, 0)
    if v1 is not None:
        score(v1[0], "temporal", "care_v1_static")
        score(E.rolling(v1[0], y, ym, warm=v1[0].attrs["val"]), "temporal", "care_v1")
        score(dynamic(v1[0], y, ym, extra, warm=v1[0].attrs["val"], **cfg), "temporal", "care_v1+dyn")
    members = [m for m in SECOND if (f"{m}@prof", 0) in Pt and os.path.exists(f"{NEST}/temporal/{m}@prof.parquet")]
    notes["temporal_members"] = members
    print("temporal members with nested predictions:", members, flush=True)
    if members:
        test = {m: Pt[(f"{m}@prof", 0)] for m in members}
        nested = {m: pd.read_parquet(f"{NEST}/temporal/{m}@prof.parquet").assign(outer=0) for m in members}
        fr, hold, info = second_layer(f, y, test, nested, members, with_year=False)
        score(fr["stack"], "temporal", "stack_static")
        score(fr["mcboost"], "temporal", "stack+mcboost_static")
        score(fr["hicare"], "temporal", "hicare_static")
        score(E.rolling(fr["hicare"], y, ym, warm=hold), "temporal", "hicare+roll")
        score(dynamic(fr["hicare"], y, ym, extra, warm=hold, **cfg), "temporal", "hicare")

    # ---------------- geographic -------------------------------------------------------------
    Pt, Vt = E.load_dir(E.PRED, "geo", "total"), E.load_dir(E.PVAL, "geo", "total")
    if ("paper_lr", 0) in Pt:
        score(Pt[("paper_lr", 0)], "geo", "reference")
    for nm, tag in (("care_v1", "@no_geo"), ("hicare", "@nogeo_prof"), ("full_with_profiles", "@prof")):
        pool = MEMBERS if nm == "care_v1" else SECOND
        mem = [f"{m}{tag}" for m in pool if (f"{m}{tag}", 0) in Pt and (f"{m}{tag}", 0) in Vt]
        if mem:
            r = E.ensemble(Pt, Vt, y, mem, 0)
            score(r[0], "geo", nm)
            notes[f"geo_{nm}_members"] = mem
    for m in MEMBERS:
        for tag in ("@no_geo", "@nogeo_prof"):
            if (f"{m}{tag}", 0) in Pt:
                score(Pt[(f"{m}{tag}", 0)], "geo", f"{m}{tag}")
    notes["second_layer_members"] = SECOND

    L = pd.DataFrame(long)
    L.to_csv(f"{TAB}/v2_metrics.csv", index=False)
    pd.concat(subs).to_csv(f"{TAB}/v2_subgroups.csv", index=False)
    json.dump(notes, open(f"{TAB}/v2_notes.json", "w"), indent=1, default=str)

    # win / loss of HiCARE against the reference, 13 metrics + 4 subgroup-calibration metrics
    W = []
    piv = L.pivot_table(index=["protocol", "sample", "metric"], columns="model", values="value")
    extra_m = {"sub_max_abs_log_oe": False, "sub_mean_abs_log_oe": False, "sub_max_ece": False,
               "sub_mean_ece": False}
    for (p, s, m), r in piv.iterrows():
        hb = MT.HIGHER.get(m, extra_m.get(m))
        if hb is None or "hicare" not in r or "reference" not in r or pd.isna(r["hicare"]) or pd.isna(r["reference"]):
            continue
        W.append(dict(protocol=p, sample=s, metric=m, hicare=r["hicare"], reference=r["reference"],
                      care_v1=r.get("care_v1", np.nan),
                      hicare_wins=bool((r["hicare"] > r["reference"]) == hb),
                      v1_wins=bool((r.get("care_v1", np.nan) > r["reference"]) == hb) if pd.notna(r.get("care_v1", np.nan)) else None))
    W = pd.DataFrame(W)
    W.to_csv(f"{TAB}/v2_winloss.csv", index=False)
    core = W[W.metric.isin(list(MT.HIGHER))]
    print(core.groupby(["protocol", "sample"]).agg(hicare=("hicare_wins", "sum"), v1=("v1_wins", "sum"),
                                                   n=("metric", "size")).to_string())
    print("13-metric tally  HiCARE:", int(core.hicare_wins.sum()), "/", len(core),
          " | v1:", int(core.v1_wins.fillna(False).sum()))
    print(core[~core.hicare_wins][["protocol", "sample", "metric", "hicare", "reference", "care_v1"]].round(5).to_string(index=False))
    show = L[L.metric.isin(["auroc", "auprc", "logloss", "ici", "cal_int", "cal_slope", "sub_max_abs_log_oe",
                            "sub_mean_abs_log_oe", "sub_max_ece"])]
    print(show[show["sample"] == "total"].pivot_table(index=["protocol", "model"], columns="metric", values="value")
          .round(4).to_string())
    print(show[show["sample"] != "total"].pivot_table(index=["protocol", "sample", "model"], columns="metric",
                                                     values="value").round(4).to_string())

    if "--boot" in sys.argv:
        rows = []
        for p in ("cv", "temporal", "geo"):
            fa, fb = f"{OUTP}/{p}/hicare.parquet", f"{OUTP}/{p}/reference.parquet"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            a = pd.read_parquet(fa).set_index("idx").sort_index()
            b = pd.read_parquet(fb).set_index("idx").sort_index()
            for s, ids in strata.items():
                ii = a.index.intersection(ids)
                for r in E.bootstrap(y.loc[ii].values, a.p.loc[ii].values, b.p.loc[ii].values, s, B=300):
                    rows.append(dict(protocol=p, sample=s, **r))
        pd.DataFrame(rows).to_csv(f"{TAB}/v2_boot.csv", index=False)
        B = pd.DataFrame(rows)
        sig = (B.lo > 0) | (B.hi < 0)
        print("bootstrap:", len(B), "significant", int(sig.sum()), "for HiCARE", int((sig & B.care_better).sum()),
              "for reference", int((sig & ~B.care_better).sum()))


if __name__ == "__main__":
    main()
