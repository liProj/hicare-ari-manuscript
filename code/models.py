"""Model zoo for the ARI in-hospital mortality comparison.

Every model exposes the same call:  fit_predict(name, tr, te, seed, cols=None, **kw) -> p(te)
`tr` / `te` are slices of the feature frame built by features.py (they include `dead`).

  paper_lr        the reference article's model: unpenalised logistic regression on its seven
                  categorical variables (year dummies included)
  paper_lr_carry  same, but a year the model has not seen is scored with the latest training
                  year's coefficient (the only way a year dummy can be used prospectively)
  paper_lr_noyear same, year removed
  lr_full         ridge logistic regression on the wide feature set (one-hot + age spline)
  lgbm7           LightGBM restricted to the article's seven variables
  lgbm / catboost / xgb / tabm / tabicl    wide feature set
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import cohort as C
import features as F

NTHREADS = int(os.environ.get("ARI_THREADS", "10"))
FULL = F.NUM + F.CTX + F.CAT
PAPER_LEVELS = {
    "age_group": C.AGE_LABELS, "sex": ["Male", "Female"], "ethnicity": C.ETH_LEVELS,
    "area": ["Urban", "Rural"], "sector": ["Public", "Private"], "ari_type": C.TYPE_LEVELS}
PAPER_REF = {"age_group": "20-29", "sex": "Male", "ethnicity": "Mestizo", "area": "Urban",
             "sector": "Public", "ari_type": C.TYPE_LEVELS[0]}


# (positions of the inner validation records inside the training fold, their predictions) of
# the most recent fit; run_cv.py stores them so the ensemble can be recalibrated on data that
# no member was trained on and that never includes a test record
LAST_VAL = None


def inner_split(y, seed, frac=0.1):
    sss = StratifiedShuffleSplit(1, test_size=frac, random_state=seed)
    a, b = next(sss.split(np.zeros(len(y)), y))
    return a, b


# ---------------------------------------------------------------- reference model ----------
def paper_design(df, years, year_mode="dummy", last_year=None):
    cols = []
    for v, levels in PAPER_LEVELS.items():
        s = df[v].astype(str).values
        for lv in levels:
            if lv != PAPER_REF[v]:
                cols.append((s == lv).astype(np.float32))
    if year_mode != "drop":
        y = df.year.values.copy()
        if year_mode == "carry":
            y = np.where(y > last_year, last_year, y)
        for yr in years[1:]:
            cols.append((y == yr).astype(np.float32))
    return np.column_stack(cols)


def fit_paper_lr(tr, te, year_mode="dummy"):
    from sklearn.linear_model import LogisticRegression
    years = sorted(tr.year.unique())
    Xtr = paper_design(tr, years, year_mode, years[-1])
    Xte = paper_design(te, years, year_mode, years[-1])
    keep = Xtr.std(0) > 0                       # e.g. adult age levels in the paediatric sample
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-8)
    m.fit(Xtr[:, keep], tr.dead.values)
    # fitted probabilities for the last three training months: the warm start of the rolling
    # recalibration when the model is applied forward in time (the reference has no held-out
    # validation records, so these are in-sample -- with ~40 parameters the difference is nil)
    global LAST_VAL
    tail = np.flatnonzero(tr.adm_ym.values >= tr.adm_ym.values.max() - 2)
    LAST_VAL = (tail, m.predict_proba(Xtr[tail][:, keep])[:, 1])
    return m.predict_proba(Xte[:, keep])[:, 1]


# ---------------------------------------------------------------- encoders -----------------
def cat_frames(tr, te, cols, min_count=1):
    """Pandas categoricals with levels fixed on the training fold; unseen test levels -> NaN."""
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    for c in cols:
        if Xtr[c].dtype == object or str(Xtr[c].dtype) == "string" or Xtr[c].dtype.name == "category":
            vc = Xtr[c].astype(str).value_counts()
            lv = vc.index[vc >= min_count]
            Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=lv)
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=lv)
    return Xtr, Xte


def is_cat(tr, c):
    return tr[c].dtype == object or str(tr[c].dtype) == "string" or tr[c].dtype.name == "category"


# ---------------------------------------------------------------- GBDTs --------------------
def fit_lgbm(tr, te, seed, cols, **kw):
    import lightgbm as lgb
    Xtr, Xte = cat_frames(tr, te, cols)
    y = tr.dead.values
    a, b = inner_split(y, seed)
    p = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=80,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             cat_smooth=30, cat_l2=10, max_cat_to_onehot=8, max_bin=255, num_threads=NTHREADS,
             seed=seed, verbose=-1)
    p.update(kw)
    dtr = lgb.Dataset(Xtr.iloc[a], y[a])
    dva = lgb.Dataset(Xtr.iloc[b], y[b], reference=dtr)
    m = lgb.train(p, dtr, 4000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    fit_lgbm.last = m
    global LAST_VAL
    LAST_VAL = (b, m.predict(Xtr.iloc[b], num_iteration=m.best_iteration))
    return m.predict(Xte, num_iteration=m.best_iteration)


def fit_catboost(tr, te, seed, cols, **kw):
    from catboost import CatBoostClassifier, Pool
    cats = [c for c in cols if is_cat(tr, c)]
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype(str)
        Xte[c] = Xte[c].astype(str)
    y = tr.dead.values
    a, b = inner_split(y, seed)
    m = CatBoostClassifier(iterations=4000, learning_rate=0.08, depth=8, l2_leaf_reg=5,
                           task_type="GPU", devices="0", od_type="Iter", od_wait=150,
                           border_count=128, random_seed=seed, verbose=0,
                           thread_count=NTHREADS, **kw)
    m.fit(Pool(Xtr.iloc[a], y[a], cat_features=cats),
          eval_set=Pool(Xtr.iloc[b], y[b], cat_features=cats), use_best_model=True)
    return m.predict_proba(Pool(Xte, cat_features=cats))[:, 1]


def fit_xgb(tr, te, seed, cols, **kw):
    import xgboost as xgb
    Xtr, Xte = cat_frames(tr, te, cols)
    y = tr.dead.values
    a, b = inner_split(y, seed)
    m = xgb.XGBClassifier(n_estimators=4000, learning_rate=0.05, max_depth=8,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=5.0, tree_method="hist", device="cuda",
                          enable_categorical=True, max_cat_to_onehot=8, max_bin=256,
                          early_stopping_rounds=100, random_state=seed, n_jobs=NTHREADS, **kw)
    m.fit(Xtr.iloc[a], y[a], eval_set=[(Xtr.iloc[b], y[b])], verbose=False)
    global LAST_VAL
    LAST_VAL = (b, m.predict_proba(Xtr.iloc[b])[:, 1])
    return m.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------- linear, wide -------------
def fit_lr_full(tr, te, seed, cols, **kw):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer
    cats = [c for c in cols if is_cat(tr, c)]
    nums = [c for c in cols if c not in cats]
    oh = OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist")
    blocks_tr = [oh.fit_transform(tr[cats].astype(str))]
    blocks_te = [oh.transform(te[cats].astype(str))]
    for c in nums:
        x_tr, x_te = tr[c].values.astype(float), te[c].values.astype(float)
        med = np.nanmedian(x_tr)
        miss_tr, miss_te = np.isnan(x_tr), np.isnan(x_te)
        x_tr, x_te = np.where(miss_tr, med, x_tr), np.where(miss_te, med, x_te)
        if c == "age_cont":
            sp = SplineTransformer(n_knots=12, degree=3, knots="quantile",
                                   extrapolation="constant")
            blocks_tr.append(sparse.csr_matrix(sp.fit_transform(x_tr[:, None])))
            blocks_te.append(sparse.csr_matrix(sp.transform(x_te[:, None])))
        else:
            mu, sd = x_tr.mean(), x_tr.std() + 1e-9
            blocks_tr.append(sparse.csr_matrix(np.column_stack([(x_tr - mu) / sd, miss_tr])))
            blocks_te.append(sparse.csr_matrix(np.column_stack([(x_te - mu) / sd, miss_te])))
    Xtr, Xte = sparse.hstack(blocks_tr).tocsr(), sparse.hstack(blocks_te).tocsr()
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400)
    m.fit(Xtr, tr.dead.values)
    return m.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------- neural / in-context ------
def _encode_nn(tr, te, cols):
    cats = [c for c in cols if is_cat(tr, c)]
    nums = [c for c in cols if c not in cats]
    Xc_tr = np.zeros((len(tr), len(cats)), dtype=np.int64)
    Xc_te = np.zeros((len(te), len(cats)), dtype=np.int64)
    cards = []
    for j, c in enumerate(cats):
        vc = tr[c].astype(str).value_counts()
        lv = vc.index[vc >= 20]
        mp = {v: i + 1 for i, v in enumerate(lv)}              # 0 = rare / unseen
        Xc_tr[:, j] = tr[c].astype(str).map(mp).fillna(0).astype(np.int64).values
        Xc_te[:, j] = te[c].astype(str).map(mp).fillna(0).astype(np.int64).values
        cards.append(len(lv) + 1)
    ntr, nte = [], []
    for c in nums:
        x_tr, x_te = tr[c].values.astype(np.float32), te[c].values.astype(np.float32)
        med = np.nanmedian(x_tr)
        m_tr, m_te = np.isnan(x_tr), np.isnan(x_te)
        x_tr, x_te = np.where(m_tr, med, x_tr), np.where(m_te, med, x_te)
        mu, sd = x_tr.mean(), x_tr.std() + 1e-6
        if np.unique(x_tr[:200000]).size > 1:      # a constant column (e.g. COVID-19 share in
            ntr.append((x_tr - mu) / sd)           # a pre-2020 training window) carries nothing
            nte.append((x_te - mu) / sd)
        if 0 < m_tr.sum() < len(m_tr):
            ntr.append(m_tr.astype(np.float32))
            nte.append(m_te.astype(np.float32))
    return (np.column_stack(ntr).astype(np.float32), Xc_tr,
            np.column_stack(nte).astype(np.float32), Xc_te, cards)


def fit_tabm(tr, te, seed, cols, **kw):
    from tabm_model import train_predict
    Xn_tr, Xc_tr, Xn_te, Xc_te, cards = _encode_nn(tr, te, cols)
    y = tr.dead.values.astype(np.float32)
    a, b = inner_split(y, seed)
    p, pv = train_predict(Xn_tr[a], Xc_tr[a], y[a], Xn_tr[b], Xc_tr[b], y[b], Xn_te, Xc_te,
                          cards, seed, **kw)
    global LAST_VAL
    LAST_VAL = (b, pv)
    return p


def fit_tabicl(tr, te, seed, cols, n_context=30000, n_estimators=4, **kw):
    """TabICLv2 as a prior-fitted (meta-learned) in-context learner.  The whole training fold
    does not fit in context, so the context is a stratified random subsample of it."""
    import torch
    from tabicl import TabICLClassifier
    cats = [c for c in cols if is_cat(tr, c)]
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    for c in cats:                                   # ordinal codes, levels fixed on train
        lv = pd.Categorical(Xtr[c].astype(str)).categories
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=lv).codes.astype(np.float32)
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=lv).codes.astype(np.float32)
    y = tr.dead.values
    if len(tr) > n_context:
        sss = StratifiedShuffleSplit(1, train_size=n_context, random_state=seed)
        idx, _ = next(sss.split(np.zeros(len(y)), y))
    else:
        idx = np.arange(len(y))
    clf = TabICLClassifier(n_estimators=n_estimators, device="cuda", random_state=seed, **kw)
    clf.fit(Xtr.iloc[idx].values.astype(np.float32), y[idx])
    out = []
    for s in range(0, len(Xte), 20000):
        out.append(clf.predict_proba(Xte.iloc[s:s + 20000].values.astype(np.float32))[:, 1])
    del clf
    torch.cuda.empty_cache()
    return np.concatenate(out)


def fit_predict(name, tr, te, seed, cols=None, **kw):
    global LAST_VAL
    LAST_VAL = None
    cols = cols or FULL
    if name == "paper_lr":
        return fit_paper_lr(tr, te, "dummy")
    if name == "paper_lr_carry":
        return fit_paper_lr(tr, te, "carry")
    if name == "paper_lr_noyear":
        return fit_paper_lr(tr, te, "drop")
    if name == "lgbm7":
        return fit_lgbm(tr, te, seed, C.PAPER_VARS, num_leaves=63)
    fn = dict(lgbm=fit_lgbm, catboost=fit_catboost, xgb=fit_xgb, lr_full=fit_lr_full,
              tabm=fit_tabm, tabicl=fit_tabicl)[name]
    return fn(tr, te, seed, cols, **kw)
