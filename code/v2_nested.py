"""Nested out-of-fold predictions: the training data of the second layer (stacking,
multicalibration), produced with separation of direct model fitting from the outer test fold.
Historical context is precomputed and may include earlier outcomes from held-out records;
this is not complete isolation of historical label information.

cv        for outer test fold k and inner fold j != k, the member is trained on the three
          folds other than j and k and predicts fold j.  The four inner predictions together
          cover the whole outer training set and come from models that have not seen fold k.
temporal  5-fold out-of-fold predictions inside the 2015-2021 training period.

Each (outer, inner) piece is written separately, so an interrupted run resumes where it stopped.
Output: results/v2/nested/<protocol>/<model>@<variant>.parquet   (outer, idx, p)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import features as F
import models as M
from run_cv import feature_cols

JD = F.JD
OUT = f"{JD}/results/v2/nested"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="cv")
    ap.add_argument("--models", required=True)
    ap.add_argument("--variant", default="prof")
    a = ap.parse_args()
    f = F.build()
    f["idx"] = np.arange(len(f))
    cols = feature_cols(a.variant)
    pos = np.arange(len(f))
    skf = StratifiedKFold(5, shuffle=True, random_state=1000)        # = run_cv seed 0
    if a.protocol == "cv":
        fold = np.zeros(len(f), int)
        for k, (_, b) in enumerate(skf.split(pos, f.dead.values)):
            fold[b] = k
        jobs = [(k, j, pos[(fold != k) & (fold != j)], pos[fold == j])
                for k in range(5) for j in range(5) if j != k]
    else:
        ptr = pos[f.year.values <= 2021]
        fold = np.full(len(f), -1)
        for j, (_, b) in enumerate(skf.split(ptr, f.dead.values[ptr])):
            fold[ptr[b]] = j
        jobs = [(-1, j, ptr[fold[ptr] != j], ptr[fold[ptr] == j]) for j in range(5)]
    for name in a.models.split(","):
        tag = f"{name}@{a.variant}"
        pdir = f"{OUT}/{a.protocol}/parts"
        os.makedirs(pdir, exist_ok=True)
        final = f"{OUT}/{a.protocol}/{tag}.parquet"
        if os.path.exists(final):
            print("cached", final, flush=True)
            continue
        parts = []
        for k, j, tr, te in jobs:
            pp = f"{pdir}/{tag}_k{k}_j{j}.parquet"
            if not os.path.exists(pp):
                t0 = time.time()
                p = M.fit_predict(name, f.iloc[tr], f.iloc[te], 7000 + 10 * (k + 1) + j, cols=cols)
                pd.DataFrame(dict(outer=k, idx=f.idx.values[te], p=p.astype(np.float64))).to_parquet(
                    pp + ".tmp", index=False)
                os.replace(pp + ".tmp", pp)
                print(f"nested {a.protocol} {tag} outer {k} inner {j}: n_tr {len(tr):,} "
                      f"{time.time() - t0:.0f}s", flush=True)
            parts.append(pd.read_parquet(pp))
        pd.concat(parts, ignore_index=True).to_parquet(final + ".tmp", index=False)
        os.replace(final + ".tmp", final)
        print("wrote", final, flush=True)


if __name__ == "__main__":
    main()
