"""Choose the rolling-recalibration window and update type WITHOUT touching 2022-2023.

Development data: forward-chaining predictions of the LightGBM member for test years 2018-2021
(each year predicted by a model trained on all earlier years).  Candidates: look-back of 1, 2,
3 or 6 months x update type ("auto" = intercept and slope when the window holds >= 200 deaths,
else intercept only; "intercept" = intercept only).  The candidate with the lowest pooled log
loss is written to results/tables/rolling_config.json and used unchanged for every model,
including the reference, in the temporal test (2022-2023) and in forward chaining.

Output: results/tables/rolling_selection.csv, rolling_config.json
"""
import json

import numpy as np
import pandas as pd

import evaluate as E
import features as F
import metrics as MT

f = F.build()
y = f.dead.astype(float)
ym = f.adm_ym.values
P = E.load_dir(E.PRED, "forward", "total")
V = E.load_dir(E.PVAL, "forward", "total")
d, v = P[("lgbm", 0)], V[("lgbm", 0)]
dev = d[d.fold <= 2021]
rows = []
base = MT.all_metrics(y.loc[dev.index].values, dev.p.values, "total")
rows.append(dict(window=0, mode="none", logloss=base["logloss"], brier=base["brier"],
                 cal_int=base["cal_int"], cal_slope=base["cal_slope"], ici=base["ici"]))
for w in (1, 2, 3, 6):
    for mode in ("auto", "intercept"):
        r = E.rolling(dev, y, ym, window=w, warm=v, mode=mode)
        mt = MT.all_metrics(y.loc[r.index].values, r.p.values, "total")
        rows.append(dict(window=w, mode=mode, logloss=mt["logloss"], brier=mt["brier"],
                         cal_int=mt["cal_int"], cal_slope=mt["cal_slope"], ici=mt["ici"]))
R = pd.DataFrame(rows)
R.to_csv(f"{E.TAB}/rolling_selection.csv", index=False)
best = R[R.window > 0].sort_values("logloss").iloc[0]
json.dump(dict(window=int(best.window), mode=best["mode"],
               selected_on="forward-chaining LightGBM, test years 2018-2021"),
          open(E.ROLL_CFG, "w"))
print(R.round(5).to_string(index=False))
print("selected:", int(best.window), best["mode"])
