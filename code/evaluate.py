"""Build the ensemble, score every prediction file, and compare against the reference model.

CARE (Context-Aware Risk Ensemble)
  * three members -- LightGBM, XGBoost, TabM -- trained ONCE on the pooled (all-ages) sample
    with the wide discharge-abstract feature set;
  * member probabilities are averaged with equal weight (no fitted stacking weight);
  * the average is recalibrated with a two-parameter logistic map fitted on the members'
    inner validation records -- records held out of member training for early stopping, never
    part of a test fold;
  * paediatric and adult predictions are the pooled model's predictions for those records;
  * when predictions are made forward in time (temporal and forward-chaining protocols) the
    recalibration map is updated every month from the outcomes of the three preceding months
    ("care-roll"); the reference model is given the same update.

The reference is the article's logistic model, refitted inside each sample exactly as published.

Outputs (results/tables/): metrics_long.csv, metrics_summary.csv, bootstrap_vs_paper.csv,
winloss.csv, temporal_reference_choice.csv, recalibration.csv;
ensemble predictions in results/ensemble/<protocol>/<sample>/<name>_s<seed>.parquet
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit

import features as F
import metrics as MT

JD = F.JD
PRED = f"{JD}/results/preds"
PVAL = f"{JD}/results/preds_val"
ENS = f"{JD}/results/ensemble"
TAB = f"{JD}/results/tables"
BASE = ["lgbm", "xgb", "tabm"]
HEAD = list(MT.HIGHER)          # the 13 metrics that enter the win/loss tally
REF = {"cv": "paper_lr", "temporal": "paper_lr_best", "forward": "paper_lr_best",
       "geo": "paper_lr"}
# the model that carries the claim under each protocol: when predictions are made forward in
# time, CARE includes its prospective rolling recalibration
MAIN = {"cv": "care", "geo": "care-portable", "temporal": "care-roll", "forward": "care-roll"}
ROLL_CFG = f"{JD}/results/tables/rolling_config.json"


def roll_cfg():
    """Window (months) and update type of the rolling recalibration.  Chosen once by
    select_rolling.py on forward-chaining test years up to 2021 -- never on 2022-2023 -- and
    read from disk; the default (3 months, the context covariates' look-back) applies only
    until that selection has been run."""
    import json
    if os.path.exists(ROLL_CFG):
        c = json.load(open(ROLL_CFG))
        return int(c["window"]), c["mode"]
    return 3, "auto"
PROTOCOLS = ("cv", "temporal", "forward", "geo")
SAMPLES = ("total", "pediatric", "adult")


def load_dir(root, protocol, sample):
    out = {}
    for fp in sorted(glob.glob(f"{root}/{protocol}/{sample}/*.parquet")):
        name, seed = os.path.basename(fp)[:-8].rsplit("_s", 1)
        d = pd.read_parquet(fp)
        out[(name, int(seed))] = d.set_index("idx").sort_index()
    return out


def zmean(frames, how="prob"):
    """Logit of the ensemble prediction.  "prob" averages member probabilities, which keeps
    the ensemble's mean prediction equal to the members' mean; "logit" averages log-odds,
    which pulls very small probabilities down whenever members disagree (Jensen) and was found
    to under-predict paediatric deaths by about 10% -- it is kept as an ablation."""
    if how == "logit":
        return np.mean([logit(MT.clip(d.p.values)) for d in frames], axis=0)
    return logit(MT.clip(np.mean([d.p.values for d in frames], axis=0)))


def ensemble(P, V, y, members, seed, how="prob"):
    """Equal-weight mean of `members`, recalibrated fold by fold on inner-validation records.
    Returns (calibrated, raw, [(fold, a, b)]) or None if a member is missing."""
    if not all((m, seed) in P and (m, seed) in V for m in members):
        return None
    te = [P[(m, seed)] for m in members]
    va = [V[(m, seed)] for m in members]
    idx = te[0].index
    assert all(d.index.equals(idx) for d in te)
    fold = te[0].fold.values
    z = zmean(te, how)
    out = np.zeros(len(idx))
    pars = []
    for k in np.unique(fold):
        vk = [d[d.fold == k] for d in va]
        assert all(v.index.equals(vk[0].index) for v in vk), "members disagree on inner split"
        a, b = MT.platt(zmean(vk, how), y.loc[vk[0].index].values)
        out[fold == k] = expit(a + b * z[fold == k])
        pars.append((k, a, b))
    cal = pd.DataFrame(dict(fold=fold, p=out), index=idx)
    raw = pd.DataFrame(dict(fold=fold, p=expit(z)), index=idx)
    vparts = []
    for k, a, b in pars:                      # the same map applied to the validation records
        vk = [d[d.fold == k] for d in va]
        vparts.append(pd.DataFrame(dict(fold=k, p=expit(a + b * zmean(vk, how))),
                                   index=vk[0].index))
    cal.attrs["val"] = pd.concat(vparts)
    return cal, raw, pars


def rolling(d, y, ym, window=None, warm=None, mode=None):
    """Prospective rolling recalibration for predictions made forward in time.

    For each calendar month of admission t in a test fold, the logistic recalibration map is
    fitted on that fold's records admitted in months t-window..t-1 -- outcomes that have been
    observed by month t -- and applied to month t.  `warm` holds the same model's predictions
    for held-out records of the training period (the members' inner-validation records; for
    the reference model its fitted values for the last three training months), so the window
    is not empty in the first test months.  Intercept and slope are updated when the window
    holds at least 200 deaths, the intercept alone when it holds 30-199, nothing otherwise.
    No record contributes to its own recalibration and nothing after month t is used.
    """
    cw, cm = roll_cfg()
    window = window or cw
    mode = mode or cm
    out = d.copy()
    z = logit(MT.clip(d.p.values))
    yy = y.loc[d.index].values
    m = ym[d.index.values]
    newp = d.p.values.copy()
    for k in np.unique(d.fold.values):
        fk = d.fold.values == k
        zs, ys, ms = z[fk], yy[fk], m[fk]
        if warm is not None:
            wk = warm[warm.fold == k]
            zs = np.r_[zs, logit(MT.clip(wk.p.values))]
            ys = np.r_[ys, y.loc[wk.index].values]
            ms = np.r_[ms, ym[wk.index.values]]
        for t in np.unique(m[fk]):
            cur = fk & (m == t)
            win = (ms >= t - window) & (ms < t)
            ev = ys[win].sum()
            if mode == "auto" and ev >= 200 and (win.sum() - ev) >= 200:
                a, b = MT.platt(zs[win], ys[win])
                newp[cur] = expit(a + b * z[cur])
            elif ev >= 30:
                a = 0.0
                for _ in range(50):
                    mu = expit(a + zs[win])
                    step = np.sum(ys[win] - mu) / max(np.sum(mu * (1 - mu)), 1e-12)
                    a += step
                    if abs(step) < 1e-9:
                        break
                newp[cur] = expit(a + z[cur])
    out["p"] = newp
    return out


def stacked(P, y, seed=0):
    """Ablation: cross-fitted logistic stacking instead of the equal-weight mean."""
    from sklearn.linear_model import LogisticRegression
    if not all((b, seed) in P for b in BASE):
        return None
    idx = P[("lgbm", seed)].index
    Z = np.column_stack([logit(MT.clip(P[(b, seed)].p.values)) for b in BASE])
    fold = P[("lgbm", seed)].fold.values
    yy = y.loc[idx].values
    out = np.zeros(len(idx))
    for k in np.unique(fold):
        m = LogisticRegression(C=1e6, max_iter=500).fit(Z[fold != k], yy[fold != k])
        out[fold == k] = m.predict_proba(Z[fold == k])[:, 1]
    return pd.DataFrame(dict(fold=fold, p=out), index=idx)


def best_reference(P, y, ym, V=None):
    """A year dummy has no coefficient for a year the model has not seen.  The reference is
    scored three ways -- unseen year treated as the reference year, as the latest training
    year, or year removed -- each with and without the same rolling recalibration CARE uses,
    and CARE is compared with whichever of the six has the lowest test log loss, i.e. the
    reading most favourable to the reference."""
    V = V or {}
    for m in ("paper_lr", "paper_lr_carry", "paper_lr_noyear"):
        if (m, 0) in P:
            P[(m + "-roll", 0)] = rolling(P[(m, 0)], y, ym, warm=V.get((m, 0)))
    cands = [m for m in ("paper_lr", "paper_lr_carry", "paper_lr_noyear", "paper_lr-roll",
                         "paper_lr_carry-roll", "paper_lr_noyear-roll") if (m, 0) in P]
    if not cands:
        return None
    ll = {m: MT.logloss(y.loc[P[(m, 0)].index].values, P[(m, 0)].p.values) for m in cands}
    best = min(ll, key=ll.get)
    for s in (0, 1, 2):
        P[("paper_lr_best", s)] = P[(best, 0)]
    return best


def _boot(y, pa, pb, sample, seed, n_rep):
    rs = np.random.RandomState(seed)
    out = []
    n = len(y)
    for _ in range(n_rep):
        i = rs.randint(0, n, n)
        a, b = MT.all_metrics(y[i], pa[i], sample), MT.all_metrics(y[i], pb[i], sample)
        out.append({k: a[k] - b[k] for k in HEAD})
    return out


def bootstrap(y, pa, pb, sample, B=500, jobs=10):
    per = B // jobs
    res = Parallel(n_jobs=jobs)(delayed(_boot)(y, pa, pb, sample, 100 + j, per)
                                for j in range(jobs))
    D = pd.DataFrame([r for part in res for r in part])
    a, b = MT.all_metrics(y, pa, sample), MT.all_metrics(y, pb, sample)
    rows = []
    for k in HEAD:
        d = D[k].values
        better = d > 0 if MT.HIGHER[k] else d < 0
        rows.append(dict(metric=k, care=a[k], reference=b[k], diff=a[k] - b[k],
                         lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                         p_boot=float(min(1.0, 2 * min(better.mean(), 1 - better.mean())
                                          + 1.0 / len(d))),
                         care_better=bool((a[k] > b[k]) == MT.HIGHER[k])))
    return rows


def save(d, protocol, sample, name, seed):
    os.makedirs(f"{ENS}/{protocol}/{sample}", exist_ok=True)
    d = d.copy()
    d.attrs = {}                  # the attached validation frame is not parquet metadata
    d.index.name = "idx"          # an index intersection (stratum subset) drops the name
    d.reset_index().to_parquet(f"{ENS}/{protocol}/{sample}/{name}_s{seed}.parquet", index=False)


def main():
    do_boot = "--no-boot" not in sys.argv
    os.makedirs(TAB, exist_ok=True)
    f = F.build()
    y = f.dead.astype(float)
    strata = {"pediatric": f.index[f.pediatric == 1], "adult": f.index[f.pediatric == 0]}
    ym = f.adm_ym.values
    long, boots, notes, recal = [], [], [], []

    def score(P, protocol, sample):
        for (m, s), d in P.items():
            r = MT.all_metrics(y.loc[d.index].values, d.p.values, sample)
            long.extend(dict(protocol=protocol, sample=sample, model=m, seed=s, metric=k,
                             value=v) for k, v in r.items())

    for protocol in PROTOCOLS:
        Pt = load_dir(PRED, protocol, "total")
        if not Pt:
            continue
        Vt = load_dir(PVAL, protocol, "total")
        E = {}                                         # ensemble frames on the pooled sample
        for s in (0, 1, 2):
            r = ensemble(Pt, Vt, y, BASE, s)
            if r is None:
                continue
            E[("care", s)], E[("care-raw", s)], pars = r
            recal += [dict(protocol=protocol, seed=s, fold=k, a=a, b=b) for k, a, b in pars]
            if s == 0:
                E[("care-logit", s)] = ensemble(Pt, Vt, y, BASE, s, how="logit")[0]
                for drop in BASE:                      # leave-one-member-out
                    rr = ensemble(Pt, Vt, y, [b for b in BASE if b != drop], s)
                    E[(f"care-no-{drop}", s)] = rr[0]
                for solo in BASE:                      # one recalibrated member on its own
                    E[(f"{solo}-cal", s)] = ensemble(Pt, Vt, y, [solo], s)[0]
                if protocol == "cv":
                    E[("care-stacked", s)] = stacked(Pt, y, s)
            if protocol in ("temporal", "forward"):
                E[("care-roll", s)] = rolling(E[("care", s)], y, ym,
                                              warm=E[("care", s)].attrs["val"])
                E[("care-roll-cold", s)] = rolling(E[("care", s)], y, ym)
                if s == 0:
                    for solo in BASE:
                        E[(f"{solo}-roll", s)] = rolling(Pt[(solo, s)], y, ym,
                                                        warm=Vt[(solo, s)])
            if protocol == "geo" and s == 0:           # location-free configuration
                rr = ensemble(Pt, Vt, y, [b + "@no_geo" for b in BASE], s)
                if rr is not None:
                    E[("care-portable", s)] = rr[0]
        best = None if protocol in ("cv", "geo") else best_reference(Pt, y, ym, Vt)
        for sample in SAMPLES:
            if sample == "total":
                P = dict(Pt)
                P.update(E)
            else:
                P = load_dir(PRED, protocol, sample)
                if protocol not in ("cv", "geo"):
                    best_s = best_reference(P, y, ym, load_dir(PVAL, protocol, sample))
                    notes.append(dict(protocol=protocol, sample=sample, best_reference=best_s))
                ids = strata[sample]
                for key, d in E.items():               # pooled model restricted to the stratum
                    P[key] = d.loc[d.index.intersection(ids)]
                for (m, s), d in Pt.items():
                    if m in BASE or m.startswith("lgbm@"):
                        P[(m, s)] = d.loc[d.index.intersection(ids)] if (m, s) not in P else P[(m, s)]
                Vs = load_dir(PVAL, protocol, sample)
                r = ensemble(load_dir(PRED, protocol, sample), Vs, y, BASE, 0)
                if r is not None:
                    P[("care-stratum", 0)] = r[0]
            if sample == "total" and best is not None:
                notes.append(dict(protocol=protocol, sample=sample, best_reference=best))
            score(P, protocol, sample)
            for key in P:
                if key[0].startswith("care") or key[0].endswith(("-cal", "-roll")) \
                        or key[0] == "paper_lr_best":
                    save(P[key], protocol, sample, key[0], key[1])
            ref, main = REF[protocol], MAIN[protocol]
            if do_boot and (main, 0) in P and (ref, 0) in P:
                a, b = P[(main, 0)], P[(ref, 0)]
                assert a.index.equals(b.index), (protocol, sample, len(a), len(b))
                for r in bootstrap(y.loc[a.index].values, a.p.values, b.p.values, sample):
                    boots.append(dict(protocol=protocol, sample=sample, **r))
            print(protocol, sample, "models:", sorted({m for m, _ in P}), flush=True)

    L = pd.DataFrame(long)
    L.to_csv(f"{TAB}/metrics_long.csv", index=False)
    S = (L.groupby(["protocol", "sample", "model", "metric"]).value
          .agg(["mean", "std", "count"]).reset_index())
    S.to_csv(f"{TAB}/metrics_summary.csv", index=False)
    pd.DataFrame(notes).to_csv(f"{TAB}/temporal_reference_choice.csv", index=False)
    pd.DataFrame(recal).to_csv(f"{TAB}/recalibration.csv", index=False)
    if boots:
        pd.DataFrame(boots).to_csv(f"{TAB}/bootstrap_vs_paper.csv", index=False)

    W = []
    piv = S.pivot_table(index=["protocol", "sample", "metric"], columns="model", values="mean")
    for (protocol, sample, metric), r in piv.iterrows():
        ref, main = REF[protocol], MAIN[protocol]
        if metric not in HEAD or main not in r or ref not in r or pd.isna(r[main]) \
                or pd.isna(r[ref]):
            continue
        win = (r[main] > r[ref]) if MT.HIGHER[metric] else (r[main] < r[ref])
        W.append(dict(protocol=protocol, sample=sample, metric=metric, care=r[main],
                      reference=r[ref], care_wins=bool(win)))
    W = pd.DataFrame(W)
    W.to_csv(f"{TAB}/winloss.csv", index=False)
    if len(W):
        print(W.groupby(["protocol", "sample"]).care_wins.agg(["sum", "count"]).to_string())
        print("overall:", int(W.care_wins.sum()), "/", len(W))
        print(W[~W.care_wins].round(5).to_string(index=False))
    show = S[S.metric.isin(["auroc", "auprc", "brier", "logloss", "ici", "cal_int", "cal_slope"])
             & S.model.isin(["paper_lr", "paper_lr_best", "lgbm7", "lr_full", "lgbm", "xgb",
                             "tabm", "tabicl", "care", "care-raw", "care-stacked",
                             "care-stratum", "care-roll", "lgbm-roll", "paper_lr-roll",
                             "paper_lr_noyear-roll", "paper_lr_carry-roll", "student_soft",
                             "student_hard"])]
    print(show.pivot_table(index=["protocol", "sample", "model"], columns="metric",
                           values="mean").round(4).to_string())


if __name__ == "__main__":
    main()
