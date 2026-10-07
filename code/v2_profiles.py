"""Provider profiles learned from the 9.4 million NON-ARI discharges of the same registry.

The first model knew a facility only by an identifier, which says nothing about a facility it
has not seen.  Here every facility group (hospital canton x class x managing entity x area) is
described by what it does with its other patients: for an ARI admission in calendar month t the
profile summarises the group's non-ARI discharges in months t-12 .. t-1.

  prof_n          log mean monthly non-ARI discharges
  prof_mort       all-cause in-hospital death proportion
  prof_mort48     share of those deaths that occurred within 48 h of admission
  prof_los        mean length of stay (days, capped at 60)
  prof_ped        share of patients aged 0-19
  prof_old        share of patients aged >= 65
  prof_outcanton  share of patients living in another canton (referral reach)
  prof_ch_*       case-mix: share of discharges in 11 ICD-10 chapter groups

No ARI record and no ARI outcome enters a profile, so it is available for a facility in a
province the mortality model has never seen.  Output: data/cohort/profiles.parquet (fac, ym).
"""
import os

import numpy as np
import pandas as pd

import cohort as C
from build_cohort import ari_mask

JD = C.JD
OUT = f"{JD}/data/cohort/profiles.parquet"
CHAPTERS = {"infect": "AB", "neoplasm": "CD", "endocrine": "E", "mental_neuro": "FGH",
            "circulatory": "I", "respiratory_other": "J", "digestive": "K", "genitourinary": "N",
            "obstetric": "O", "perinatal": "PQ", "injury": "ST"}
PROF = (["prof_n", "prof_mort", "prof_mort48", "prof_los", "prof_ped", "prof_old",
         "prof_outcanton"] + [f"prof_ch_{k}" for k in CHAPTERS])
WINDOW = 12


def monthly(year):
    d = pd.read_parquet(f"{JD}/data/cohort/all_{year}.parquet")
    d = d[~ari_mask(d)]
    fac = (d.cant_ubi.astype(str) + "_" + d.clase.round().astype("Int64").astype(str) + "_"
           + d.entidad.round().astype("Int64").astype(str) + "_"
           + d.area_ubi.round().astype("Int64").astype(str))
    age = np.where(d.cod_edad == 4, d.edad, 0.0)
    ch = d.causa3.fillna("").str[:1]
    g = pd.DataFrame(dict(
        fac=fac.values, ym=(d.anio_egr * 12 + d.mes_egr - 1).astype("Int64").values, n=1,
        dead=(d.con_egrpa > 1).astype(int).values, dead48=(d.con_egrpa == 2).astype(int).values,
        los=d.dia_estad.clip(0, 60).fillna(0).values, ped=(age <= 19).astype(int),
        old=(age >= 65).astype(int),
        outcanton=(d.cant_res.astype(str).values != d.cant_ubi.astype(str).values).astype(int)))
    for k, letters in CHAPTERS.items():
        g[f"ch_{k}"] = ch.isin(list(letters)).astype(int).values
    g = g.dropna(subset=["ym"])
    return g.groupby(["fac", "ym"]).sum()


def main():
    M = pd.concat([monthly(y) for y in C.YEARS]).groupby(level=[0, 1]).sum()
    print("facility-months:", len(M), "| non-ARI discharges:", int(M.n.sum()))
    ym_all = pd.RangeIndex(int(M.index.get_level_values(1).min()),
                           int(M.index.get_level_values(1).max()) + 2)
    rows = []
    for fac, g in M.groupby(level=0):
        g = g.droplevel(0).reindex(ym_all, fill_value=0)
        w = g.rolling(WINDOW, min_periods=1).sum().shift(1)      # months t-12 .. t-1
        months = (g.n > 0).rolling(WINDOW, min_periods=1).sum().shift(1)
        n = w.n.where(w.n >= 30)                                 # too few to describe a facility
        out = pd.DataFrame({
            "prof_n": np.log1p(w.n / months.clip(lower=1)).where(w.n >= 30),
            "prof_mort": w.dead / n, "prof_mort48": w.dead48 / w.dead.where(w.dead >= 10),
            "prof_los": w.los / n, "prof_ped": w.ped / n, "prof_old": w.old / n,
            "prof_outcanton": w.outcanton / n})
        for k in CHAPTERS:
            out[f"prof_ch_{k}"] = w[f"ch_{k}"] / n
        out["fac"] = fac
        out["ym"] = ym_all.values
        rows.append(out)
    P = pd.concat(rows, ignore_index=True)
    P[PROF] = P[PROF].astype("float32")
    P.to_parquet(OUT, index=False)
    print("profiles:", P.shape, "facilities:", P.fac.nunique())
    print(P[PROF].describe().T[["count", "mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
