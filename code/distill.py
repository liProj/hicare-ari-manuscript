"""Knowledge distillation: compress the tree teacher into a compact, portable logistic score.

Student ("CARE-Score") = one logistic regression that needs only what the reference article's
model needs minus the calendar-year label -- age, sex, ethnicity, area of residence, sector and
type of infection -- plus three national epidemic-context numbers anyone can compute from the
public registry (COVID-19 share, adult in-hospital death proportion and admission volume in the
previous three months).  Age enters as a restricted cubic spline and is allowed to interact
with the infection group; sector interacts with the infection group.

Three ways of fitting the same student are compared:
  student_hard   observed outcomes only
  student_soft   teacher probabilities as fractional targets (pure distillation)
  student_mix    half observed outcome, half teacher probability

The teacher is the LightGBM member trained on the pooled (all-ages) sample; students are fitted
separately for the total, paediatric and adult samples.

Leakage control.  Under K-fold CV an out-of-fold teacher prediction for a training record comes
from a teacher that has seen the test fold.  Soft labels are therefore produced by teachers that
exclude BOTH the record's own fold and the outer test fold: for outer fold k and inner fold j the
teacher is trained on the three remaining folds.  Under the temporal protocol the soft labels
are 5-fold out-of-fold predictions inside the 2015-2021 training period.

Writes results/preds/<protocol>/<sample>/student_{hard,soft,mix}_s0.parquet and the fitted
full-data student coefficients to results/tables/student_coefficients_<sample>.csv.
"""
import os
import sys

os.environ.setdefault("ARI_THREADS", "6")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import cohort as C
import features as F
import models as M

JD = F.JD
CTX3 = ["ctx_nat_covid", "ctx_nat_cfr_adult", "ctx_nat_n"]
TGROUP = {**{t: "other" for t in C.TYPE_LEVELS}, "Viral pneumonia": "pneumonia",
          "Bacterial pneumonia": "pneumonia", "Other pneumonias": "pneumonia",
          "COVID-19": "covid"}


def rcs(x, knots):
    """Restricted cubic spline basis (Harrell): linear beyond the outer knots."""
    k = np.asarray(knots, float)
    out = [x]
    kn, kn1 = k[-1], k[-2]
    for j in range(len(k) - 2):
        t = (np.clip(x - k[j], 0, None) ** 3
             - np.clip(x - kn1, 0, None) ** 3 * (kn - k[j]) / (kn - kn1)
             + np.clip(x - kn, 0, None) ** 3 * (kn1 - k[j]) / (kn - kn1))
        out.append(t / (kn - k[0]) ** 2)
    return np.column_stack(out)


class Student:
    def fit_design(self, tr):
        a = tr.age_cont.values.astype(float)
        q = [0.05, 0.25, 0.5, 0.75, 0.95] if np.unique(a).size > 30 else [0.1, 0.5, 0.9]
        self.knots = np.unique(np.quantile(a, q))
        if len(self.knots) < 3:
            self.knots = np.array([a.min(), np.median(a), a.max()]) + np.array([0, 1e-3, 2e-3])
        self.med = {c: float(np.nanmedian(tr[c].values)) for c in CTX3}
        X, names = self._raw(tr)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.keep = self.sd > 1e-9
        self.names = [n for n, k in zip(names, self.keep) if k]
        return self

    def _raw(self, df):
        cols, names = [], []
        A = rcs(df.age_cont.values.astype(float), self.knots)
        for j in range(A.shape[1]):
            cols.append(A[:, j]); names.append(f"age_s{j}")
        cols.append((df.age_cont.values < 1).astype(float)); names.append("age<1y")
        cols.append((df.age_cont.values < 28 / 365.25).astype(float)); names.append("neonate")
        for v, levels in M.PAPER_LEVELS.items():
            if v == "age_group":
                continue
            s = df[v].astype(str).values
            for lv in levels:
                if lv != M.PAPER_REF[v]:
                    cols.append((s == lv).astype(float)); names.append(f"{v}={lv}")
        g = df.ari_type.astype(str).map(TGROUP).values
        for grp in ("pneumonia", "covid"):
            m = (g == grp).astype(float)
            for j in range(A.shape[1]):
                cols.append(m * A[:, j]); names.append(f"{grp}:age_s{j}")
            cols.append(m * (df.sector.astype(str).values == "Private")); names.append(f"{grp}:private")
        cols.append((df.sex.astype(str).values == "Female") * A[:, 0]); names.append("female:age")
        for c in CTX3:
            x = df[c].values.astype(float)
            miss = np.isnan(x)
            x = np.where(miss, self.med[c], x)
            cols.append(x); names.append(c)
            cols.append((g == "covid") * x); names.append(f"covid:{c}")
        cols.append(np.isnan(df[CTX3[0]].values).astype(float)); names.append("ctx_missing")
        return np.column_stack(cols), names

    def design(self, df):
        X, _ = self._raw(df)
        return ((X - self.mu) / np.where(self.keep, self.sd, 1))[:, self.keep]


def fit_student(Xtr, y, soft, mode):
    m = LogisticRegression(C=10.0, solver="lbfgs", max_iter=2000)
    if mode == "hard":
        m.fit(Xtr, y)
        return m
    t = soft if mode == "soft" else 0.5 * soft + 0.5 * y
    X2 = np.vstack([Xtr, Xtr])
    y2 = np.r_[np.ones(len(y)), np.zeros(len(y))]
    w2 = np.r_[t, 1 - t]
    m.fit(X2, y2, sample_weight=w2)
    return m


def soft_labels(df, folds, exclude, seed):
    """Out-of-fold teacher probabilities for every record not in `exclude`."""
    soft = np.full(len(df), np.nan)
    pos = np.arange(len(df))
    for j in np.unique(folds):
        if j == exclude:
            continue
        tr = pos[(folds != j) & (folds != exclude)]
        te = pos[folds == j]
        soft[te] = M.fit_lgbm(df.iloc[tr], df.iloc[te], seed * 100 + int(j), M.FULL)
    return soft


def main():
    """Teacher = LightGBM member trained on the pooled sample.  Students are fitted per sample
    (total / paediatric / adult) on the training part of the same pooled folds."""
    protocol = sys.argv[1] if len(sys.argv) > 1 else "cv"
    df = F.build()
    df["idx"] = np.arange(len(df))
    y = df.dead.values.astype(float)
    pos = np.arange(len(df))
    masks = {"total": np.ones(len(df), bool), "pediatric": df.pediatric.values == 1,
             "adult": df.pediatric.values == 0}
    done = all(os.path.exists(f"{JD}/results/preds/{protocol}/{s}/student_soft_s0.parquet")
               for s in masks)
    if done:
        print("cached", protocol)
        return
    skf = StratifiedKFold(5, shuffle=True, random_state=1000)              # = run_cv seed 0
    if protocol == "cv":
        folds = np.zeros(len(df), int)
        for k, (_, b) in enumerate(skf.split(pos, df.dead.values)):
            folds[b] = k
        outer = [(k, folds != k, folds == k) for k in range(5)]
    else:
        trm = df.year.values <= 2021
        folds = np.full(len(df), -1)
        ptr = pos[trm]
        for k, (_, b) in enumerate(skf.split(ptr, df.dead.values[ptr])):
            folds[ptr[b]] = k
        outer = [(-1, trm, ~trm)]
    res = {s: {m: [] for m in ("hard", "soft", "mix")} for s in masks}
    from sklearn.metrics import roc_auc_score
    for k, trm_, tem_ in outer:
        if protocol == "cv":
            soft = soft_labels(df, folds, k, 0)
        else:
            soft = np.full(len(df), np.nan)
            for j in range(5):
                a, b = ptr[folds[ptr] != j], ptr[folds[ptr] == j]
                soft[b] = M.fit_lgbm(df.iloc[a], df.iloc[b], j, M.FULL)
        for sample, sm in masks.items():
            tr, te = pos[trm_ & sm], pos[tem_ & sm]
            st = Student().fit_design(df.iloc[tr])
            Xtr, Xte = st.design(df.iloc[tr]), st.design(df.iloc[te])
            for mode in res[sample]:
                m = fit_student(Xtr, y[tr], np.clip(soft[tr], 1e-6, 1 - 1e-6), mode)
                p = m.predict_proba(Xte)[:, 1]
                res[sample][mode].append(pd.DataFrame(dict(idx=df.idx.values[te], fold=k, p=p)))
            print(protocol, sample, "fold", k,
                  {m: round(roc_auc_score(y[te], res[sample][m][-1].p.values), 4)
                   for m in res[sample]}, flush=True)
    for sample in masks:
        outdir = f"{JD}/results/preds/{protocol}/{sample}"
        os.makedirs(outdir, exist_ok=True)
        for mode, parts in res[sample].items():
            pd.concat(parts, ignore_index=True).to_parquet(
                f"{outdir}/student_{mode}_s0.parquet", index=False)
    if protocol == "cv":              # coefficients of the full-data distilled student
        soft_all = np.zeros(len(df))
        for k in range(5):
            a, b = pos[folds != k], pos[folds == k]
            soft_all[b] = M.fit_lgbm(df.iloc[a], df.iloc[b], k, M.FULL)
        os.makedirs(f"{JD}/results/tables", exist_ok=True)
        for sample, sm in masks.items():
            st = Student().fit_design(df[sm])
            m = fit_student(st.design(df[sm]), y[sm], np.clip(soft_all[sm], 1e-6, 1 - 1e-6),
                            "soft")
            pd.DataFrame(dict(term=st.names, coef_std=m.coef_[0], mean=st.mu[st.keep],
                              sd=st.sd[st.keep])).assign(intercept=m.intercept_[0]).to_csv(
                f"{JD}/results/tables/student_coefficients_{sample}.csv", index=False)


if __name__ == "__main__":
    main()
