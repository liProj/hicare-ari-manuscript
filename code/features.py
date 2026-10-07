"""Feature frame for the new model (CARE: Context-Aware Risk Ensemble).

Everything here is recorded in the same public discharge abstract the reference article uses.
The reference model sees seven coarse categorical variables.  The wider set keeps their full
resolution (continuous age, nine ethnicity codes, four-character ICD-10 code, three-level sector)
and adds what the article leaves unused: where the patient lives, where and in what kind of
facility they were treated, whether they were treated outside their own province or canton, and
the calendar position of the admission.

Two groups are deliberately NOT features of the main model:
  * anything determined by the outcome -- length of stay, discharge date, discharge specialty,
    timing of death (they enter one ablation only, labelled as an unusable upper bound);
  * the calendar-year label.  A year dummy cannot be applied to a year the model has not seen.
    It is replaced by lagged epidemic-context covariates, computed for every record from
    admissions in the three calendar months BEFORE its own admission month: ARI admission
    volume, COVID-19 share and in-hospital death proportion, nationally and in the hospital's
    province, plus the admitting facility group's recent volume relative to its own past year.

The registry has no facility identifier.  `fac` = hospital canton x facility class x managing
entity x hospital area is the finest facility grouping the public file allows.
"""
import os

import numpy as np
import pandas as pd

import cohort as C

JD = C.JD
CACHE = f"{JD}/data/cohort/features.parquet"

CAT = ["sexo", "etnia", "nac_pac", "area_res", "prov_res", "cant_res", "prov_ubi", "cant_ubi",
       "area_ubi", "clase", "tipo", "entidad", "sector3", "icd3", "icd4", "ari_type_c",
       "age_unit", "mes_ingr", "adm_dow", "same_prov", "same_cant", "fac"]
NUM = ["age_cont", "dia_ingr"]
CTX = ["ctx_nat_n", "ctx_nat_covid", "ctx_nat_cfr", "ctx_nat_cfr_adult", "ctx_prov_n",
       "ctx_prov_covid", "ctx_prov_cfr", "ctx_fac_n", "ctx_fac_ratio"]
POST = ["dia_estad", "esp_egrpa"]            # outcome-dependent; ablation only
# provider profile from the facility group's NON-ARI discharges of the previous 12 months
# (v2_profiles.py): describes a facility by what it does, not by who it is
PROF = (["prof_n", "prof_mort", "prof_mort48", "prof_los", "prof_ped", "prof_old",
         "prof_outcanton"] + [f"prof_ch_{k}" for k in (
             "infect", "neoplasm", "endocrine", "mental_neuro", "circulatory",
             "respiratory_other", "digestive", "genitourinary", "obstetric", "perinatal",
             "injury")])
# clinical case-mix: what the patient brings, nothing about where they live or are treated
CLIN = ["age_cont", "age_unit", "sexo", "icd3", "icd4", "ari_type_c", "mes_ingr"]
CTX_NAT = ["ctx_nat_n", "ctx_nat_covid", "ctx_nat_cfr", "ctx_nat_cfr_adult"]
YEAR = ["year"]

# feature groups for the ablation study
GROUPS = {
    "demography": ["age_cont", "age_unit", "sexo", "etnia", "nac_pac"],
    "diagnosis": ["icd3", "icd4", "ari_type_c"],
    "residence": ["area_res", "prov_res", "cant_res"],
    "facility": ["prov_ubi", "cant_ubi", "area_ubi", "clase", "tipo", "entidad", "sector3",
                 "fac"],
    "referral": ["same_prov", "same_cant"],
    "calendar": ["mes_ingr", "adm_dow", "dia_ingr"],
    "context": CTX,
}


def _lag_window(tab, lo=1, hi=3):
    """Sum over months t-hi..t-lo for a (month-indexed) table, aligned to month t."""
    return sum(tab.shift(k, fill_value=0) for k in range(lo, hi + 1))


def _context(d):
    ym = d.adm_ym
    full = pd.RangeIndex(ym.min() - 12, ym.max() + 1)
    covid = d.ari_type.eq("COVID-19").astype(int)
    adult = (d.pediatric == 0).astype(int)
    base = pd.DataFrame(dict(ym=ym, n=1, covid=covid, dead=d.dead, adult=adult,
                             adult_dead=adult * d.dead, prov=d.prov_ubi, fac=d.fac))
    out = pd.DataFrame(index=d.index)

    nat = base.groupby("ym")[["n", "covid", "dead", "adult", "adult_dead"]].sum()
    nat = nat.reindex(full, fill_value=0)
    w = _lag_window(nat)
    hist = _lag_window(pd.Series(1, index=nat.index).where(nat.n > 0, 0))   # months with data
    nat_f = pd.DataFrame(dict(
        ctx_nat_n=np.log1p(w.n / 3), ctx_nat_covid=w.covid / w.n.where(w.n > 0),
        ctx_nat_cfr=w.dead / w.n.where(w.n > 0),
        ctx_nat_cfr_adult=w.adult_dead / w.adult.where(w.adult > 0)))
    nat_f[hist < 3] = np.nan                     # incomplete look-back at the start of 2015
    for c in nat_f:
        out[c] = ym.map(nat_f[c]).values

    def by(key, prefix, with_rate=True):
        g = base.groupby([key, "ym"])[["n", "covid", "dead"]].sum()
        res = {}
        for name in ["n", "covid", "dead"]:
            wide = g[name].unstack(key).reindex(full).fillna(0)
            res[name] = _lag_window(wide)
            if name == "n":
                res["n12"] = _lag_window(wide, 1, 12)
                res["n1"] = wide.shift(1, fill_value=0)
        short = (hist < 3).values                  # incomplete look-back at the start of 2015
        n = res["n"].astype(float)
        n.loc[short] = np.nan
        feats = {f"{prefix}_n": np.log1p(n / 3)}
        if with_rate:
            feats[f"{prefix}_covid"] = res["covid"] / n.where(n > 0)
            feats[f"{prefix}_cfr"] = res["dead"] / n.where(n >= 30)   # too few to be a rate
        else:
            ratio = res["n1"] / (res["n12"] / 12).where(res["n12"] >= 12)
            ratio.loc[short] = np.nan
            feats[f"{prefix}_ratio"] = ratio
        idx = pd.MultiIndex.from_arrays([ym.values, base[key].values])
        for name, wide in feats.items():
            s = wide.stack(future_stack=True)
            out[name] = s.reindex(idx).values
    by("prov", "ctx_prov")
    by("fac", "ctx_fac", with_rate=False)
    return out.astype("float32")


def build(force=False):
    if os.path.exists(CACHE) and not force:
        return pd.read_parquet(CACHE)
    d = C.load()
    date = pd.to_datetime(
        d.year.astype(int).astype(str) + "-" + d.mes_ingr.astype(int).astype(str).str.zfill(2)
        + "-" + d.dia_ingr.astype(int).astype(str).str.zfill(2), errors="coerce")
    d["adm_dow"] = date.dt.dayofweek.astype("float32")
    d["ari_type_c"] = d.ari_type.cat.codes.astype("float32")
    d["fac"] = (d.cant_ubi.astype(str) + "_" + d.clase.astype(int).astype(str) + "_"
                + d.entidad.astype(int).astype(str) + "_" + d.area_ubi.astype(int).astype(str))
    ctx = _context(d)
    f = pd.concat([d, ctx], axis=1)
    pp = f"{JD}/data/cohort/profiles.parquet"
    if os.path.exists(pp):
        prof = pd.read_parquet(pp).rename(columns={"ym": "adm_ym"})
        f = f.merge(prof, on=["fac", "adm_ym"], how="left", validate="m:1")
        assert len(f) == len(d)
    for c in CAT:                                   # categoricals as strings, NaN -> "NA"
        s = f[c]
        if s.dtype.kind == "f":
            s = s.round().astype("Int64")
        f[c] = s.astype("string").fillna("NA").astype(str)
    f = f.drop(columns=["adm_date"])
    f.to_parquet(CACHE, index=False)
    return f


if __name__ == "__main__":
    f = build(force=True)
    print(f.shape)
    print({c: f[c].nunique() for c in CAT})
    print(f[NUM + CTX].describe().T[["count", "mean", "min", "max"]].round(3).to_string())
    print("dow NA:", (f.adm_dow == "NA").sum())
    print(f.groupby("year")[["ctx_nat_covid", "ctx_nat_cfr", "ctx_prov_cfr", "ctx_fac_ratio"]]
          .mean().round(3).to_string())
