"""Run one model under one validation protocol on one sample; write out-of-sample predictions.

Protocols
  cv        stratified 5-fold cross-validation inside the sample (folds fixed by --seed)
  temporal  train on admissions 2015-2021, test on 2022-2023
  forward   forward chaining: for each test year 2018..2023, train on all earlier years
  geo       6-fold grouped cross-validation, hospital provinces never split across folds

Output: results/preds/<protocol>/<sample>/<model><tag>_s<seed>.parquet  (idx, fold, p)
`idx` is the row number in the feature frame, so every model is scored on identical records.
A finished file is never recomputed.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

import features as F
import models as M

JD = F.JD


def splits(df, protocol, seed):
    n = len(df)
    pos = np.arange(n)
    if protocol == "cv":
        skf = StratifiedKFold(5, shuffle=True, random_state=1000 + seed)
        for k, (a, b) in enumerate(skf.split(pos, df.dead.values)):
            yield k, a, b
    elif protocol == "temporal":
        yield 0, pos[df.year.values <= 2021], pos[df.year.values >= 2022]
    elif protocol == "forward":
        for y in range(2018, 2024):
            yield y, pos[df.year.values < y], pos[df.year.values == y]
    elif protocol == "geo":
        gkf = GroupKFold(6)
        for k, (a, b) in enumerate(gkf.split(pos, df.dead.values, df.prov_ubi.values)):
            yield k, a, b
    else:
        raise ValueError(protocol)


def feature_cols(variant):
    """Feature-set variants for the ablation study."""
    full = list(M.FULL)
    if variant in ("", "full"):
        return full
    if variant == "plus_year":
        return full + ["year"]
    if variant == "plus_post":
        return full + F.POST
    geo_ids = ["prov_res", "cant_res", "prov_ubi", "cant_ubi", "fac", "ctx_prov_n",
               "ctx_prov_covid", "ctx_prov_cfr", "ctx_fac_n", "ctx_fac_ratio"]
    if variant == "prof":        # full set + provider profile
        return full + F.PROF
    if variant == "nogeo_prof":  # location-free + provider profile: portable to unseen places
        return [c for c in full if c not in geo_ids] + F.PROF
    if variant == "clin":        # clinical case-mix + calendar period (for standardisation)
        return F.CLIN + F.CTX_NAT + ["year"]
    if variant == "clin_prov":   # clinical case-mix + period + everything about the provider
        return (F.CLIN + F.CTX_NAT + ["year", "prov_ubi", "cant_ubi", "area_ubi", "clase", "tipo",
                                      "entidad", "sector3", "fac"] + F.PROF)
    if variant == "clin_notime":
        return list(F.CLIN)
    if variant == "no_geo":      # location-free: no identifier of a place or of a facility
        drop = ["prov_res", "cant_res", "prov_ubi", "cant_ubi", "fac", "ctx_prov_n",
                "ctx_prov_covid", "ctx_prov_cfr", "ctx_fac_n", "ctx_fac_ratio"]
        return [c for c in full if c not in drop]
    if variant == "year_noctx":
        return [c for c in full if c not in F.CTX] + ["year"]
    if variant.startswith("no_"):
        drop = F.GROUPS[variant[3:]]
        return [c for c in full if c not in drop]
    if variant.startswith("only_"):
        keep = []
        for g in variant[5:].split("+"):
            keep += F.GROUPS[g]
        return keep
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="cv")
    ap.add_argument("--sample", default="total")
    ap.add_argument("--models", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="")
    ap.add_argument("--folds", default="")           # e.g. "0" to run a single fold (timing)
    ap.add_argument("--kw", default="{}")
    a = ap.parse_args()

    f = F.build()
    f["idx"] = np.arange(len(f))
    if a.sample == "pediatric":
        df = f[f.pediatric == 1].reset_index(drop=True)
    elif a.sample == "adult":
        df = f[f.pediatric == 0].reset_index(drop=True)
    else:
        df = f
    cols = feature_cols(a.variant)
    tag = f"@{a.variant}" if a.variant else ""
    outdir = f"{JD}/results/preds/{a.protocol}/{a.sample}"
    os.makedirs(outdir, exist_ok=True)
    only = {int(x) for x in a.folds.split(",")} if a.folds else None
    kw = json.loads(a.kw)

    for name in a.models.split(","):
        out = f"{outdir}/{name}{tag}_s{a.seed}.parquet"
        vout = out.replace("/results/preds/", "/results/preds_val/")
        needs_val = name in ("lgbm", "xgb", "tabm") or (
            a.protocol in ("temporal", "forward") and name.startswith("paper_lr"))
        if os.path.exists(out) and only is None and (os.path.exists(vout) or not needs_val):
            print("cached", out, flush=True)
            continue
        parts, vparts, t0 = [], [], time.time()
        for k, tr, te in splits(df, a.protocol, a.seed):
            if only is not None and k not in only:
                continue
            t1 = time.time()
            p = M.fit_predict(name, df.iloc[tr], df.iloc[te], a.seed * 100 + (k % 100),
                              cols=cols, **kw)
            y = df.dead.values[te]
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
            print(f"{a.protocol}/{a.sample}/{name}{tag} s{a.seed} fold {k}: n_tr {len(tr):,} "
                  f"n_te {len(te):,} AUROC {auc:.4f}  {time.time() - t1:.0f}s", flush=True)
            parts.append(pd.DataFrame(dict(idx=df.idx.values[te], fold=k,
                                           p=p.astype(np.float64))))
            if M.LAST_VAL is not None and needs_val:
                b, pv = M.LAST_VAL
                vparts.append(pd.DataFrame(dict(idx=df.idx.values[tr][b], fold=k,
                                                p=np.asarray(pv, dtype=np.float64))))
        res = pd.concat(parts, ignore_index=True)
        if only is None:
            if vparts:
                os.makedirs(os.path.dirname(vout), exist_ok=True)
                pd.concat(vparts, ignore_index=True).to_parquet(vout + ".tmp", index=False)
                os.replace(vout + ".tmp", vout)
            tmp = out + ".tmp"
            res.to_parquet(tmp, index=False)
            os.replace(tmp, out)
            print(f"wrote {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
