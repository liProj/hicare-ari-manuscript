"""TreeSHAP attribution for the LightGBM member of CARE (pooled sample).

A model is fitted on four of the five seed-0 folds and explained on a random 60,000-record
sample of the held-out fold with LightGBM's exact TreeSHAP (pred_contrib).

Outputs: results/tables/shap_importance.csv   feature, group, mean_abs_shap
         results/tables/shap_groups.csv       group, share of total mean |SHAP|
         results/tables/shap_age.parquet      age, infection group, SHAP(age) for the dependence plot
"""
import os

os.environ.setdefault("ARI_THREADS", "6")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import features as F
import models as M

JD = F.JD


def main():
    f = F.build()
    skf = StratifiedKFold(5, shuffle=True, random_state=1000)
    tr, te = next(skf.split(np.arange(len(f)), f.dead.values))
    rs = np.random.RandomState(0)
    te = rs.choice(te, 60000, replace=False)
    M.fit_lgbm(f.iloc[tr], f.iloc[te], 0, M.FULL)
    m = M.fit_lgbm.last
    _, Xte = M.cat_frames(f.iloc[tr], f.iloc[te], M.FULL)
    sv = m.predict(Xte, pred_contrib=True, num_iteration=m.best_iteration)[:, :-1]
    grp = {c: g for g, cols in F.GROUPS.items() for c in cols}
    imp = pd.DataFrame(dict(feature=M.FULL, group=[grp[c] for c in M.FULL],
                            mean_abs_shap=np.abs(sv).mean(0))).sort_values(
        "mean_abs_shap", ascending=False)
    os.makedirs(f"{JD}/results/tables", exist_ok=True)
    imp.to_csv(f"{JD}/results/tables/shap_importance.csv", index=False)
    g = imp.groupby("group").mean_abs_shap.sum().sort_values(ascending=False)
    (g / g.sum()).rename("share").reset_index().to_csv(f"{JD}/results/tables/shap_groups.csv",
                                                      index=False)
    sub = f.iloc[te]
    tg = np.where(sub.ari_type.astype(str) == "COVID-19", "COVID-19",
                  np.where(sub.ari_type.astype(str).str.contains("pneumonia"), "Pneumonia",
                           "Other ARI"))
    pd.DataFrame(dict(age=sub.age_cont.values, group=tg,
                      shap_age=sv[:, M.FULL.index("age_cont")],
                      shap_ctx=sv[:, [M.FULL.index(c) for c in F.CTX]].sum(1),
                      ctx_nat_covid=sub.ctx_nat_covid.values, year=sub.year.values)).to_parquet(
        f"{JD}/results/tables/shap_age.parquet", index=False)
    print(imp.head(15).to_string(index=False))
    print((g / g.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
