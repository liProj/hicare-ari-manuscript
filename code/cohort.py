"""Shared cohort definition for the Medicina 62:1752 reproduction.

`load()` returns the analysis cohort with (a) the seven variables of the reference article,
coded exactly as its Table 1, and (b) the wider admission-time feature set used by the new model.

Two decisions are needed to land on the article's 576,195 hospitalisations / 41,048 deaths; the
first is printed in the article (as a table header), the second is not printed anywhere:

  1. "Year" is the year of ADMISSION (anio_ingr), not the year of the registry file.  INEC files
     are organised by discharge year, so each file also holds patients admitted the year before;
     records admitted before 2015 are outside the study window.
  2. COVID-19 codes (U07.1/U07.2) attached to an admission year before 2020 are dropped.  There
     are 14 such records, all long-stay patients admitted 2016-2019 who died in 2020.

A third unprinted decision affects only the split of pneumonia into types (see `ari_type`).
"""
import os

import numpy as np
import pandas as pd

JD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT = f"{JD}/data/cohort/ari.parquet"

AGE_BINS = [-1, 4, 9, 14, 19, 29, 39, 49, 59, 69, 200]
AGE_LABELS = ["0-4", "5-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-59", "60-69",
              ">=70"]
ETH_MAP = {6: "Mestizo", 1: "Indigenous", 2: "Black/Afro", 3: "Black/Afro", 5: "Montubio",
           7: "White", 4: "Other", 8: "Other", 9: "Unknown"}
ETH_LEVELS = ["Mestizo", "Indigenous", "Black/Afro", "Montubio", "White", "Other", "Unknown"]
TYPE_LEVELS = ["Acute upper respiratory infections", "Influenza", "Viral pneumonia",
               "Bacterial pneumonia", "Other pneumonias", "Acute bronchitis",
               "Acute bronchiolitis", "Unspecified acute lower respiratory infection",
               "Suppurative and necrotic conditions", "COVID-19"]
YEARS = list(range(2015, 2024))
PROVINCES = {
    "01": "Azuay", "02": "Bolívar", "03": "Cañar", "04": "Carchi", "05": "Cotopaxi",
    "06": "Chimborazo", "07": "El Oro", "08": "Esmeraldas", "09": "Guayas", "10": "Imbabura",
    "11": "Loja", "12": "Los Ríos", "13": "Manabí", "14": "Morona Santiago", "15": "Napo",
    "16": "Pastaza", "17": "Pichincha", "18": "Tungurahua", "19": "Zamora Chinchipe",
    "20": "Galápagos", "21": "Sucumbíos", "22": "Orellana",
    "23": "Santo Domingo de los Tsáchilas", "24": "Santa Elena"}

# the seven predictors of the reference model, with the reference level first
PAPER_VARS = ["age_group", "sex", "ethnicity", "area", "sector", "ari_type", "year"]


def ari_type(c3, c4):
    """Ten infection types of the reference article.

    The article prints the split as J12 viral / J13-J15 bacterial / J16-J18 other, but its
    Table 1 only reconciles if the organism-specific fourth-character codes inside J16-J17 are
    moved to the organism group: J16.0 (chlamydial) and J17.0 (in bacterial diseases classified
    elsewhere) to bacterial pneumonia, J17.1 (in viral diseases classified elsewhere) to viral.
    """
    n = pd.to_numeric(c3.str[1:3], errors="coerce")
    out = pd.Series(pd.NA, index=c3.index, dtype="object")
    j = c3.str[0].eq("J")
    for lo, hi, lab in [(0, 6, 0), (9, 11, 1), (12, 12, 2), (13, 15, 3), (16, 18, 4),
                        (20, 20, 5), (21, 21, 6), (22, 22, 7), (85, 86, 8)]:
        out[j & n.between(lo, hi)] = TYPE_LEVELS[lab]
    out[c4.isin(["J160", "J170"])] = TYPE_LEVELS[3]
    out[c4.eq("J171")] = TYPE_LEVELS[2]
    out[c3.eq("U07")] = TYPE_LEVELS[9]
    return out


def load(apply_rules=True):
    a = pd.read_parquet(COHORT)
    a["covid"] = a.causa3.eq("U07")
    if apply_rules:
        a = a[a.anio_ingr.between(2015, 2023)]
        a = a[~(a.covid & (a.anio_ingr < 2020))]
    a = a.reset_index(drop=True)

    d = pd.DataFrame(index=a.index)
    d["dead"] = (a.con_egrpa > 1).astype("int8")
    d["death_lt48h"] = (a.con_egrpa == 2).astype("int8")
    # age: only cod_edad == 4 is in years; hours / days / months are all under one year
    yrs = np.where(a.cod_edad == 4, a.edad, 0.0)
    d["age"] = yrs.astype("float32")
    frac = np.select([a.cod_edad == 1, a.cod_edad == 2, a.cod_edad == 3],
                     [a.edad / 24 / 365.25, a.edad / 365.25, a.edad / 12], default=a.edad)
    d["age_cont"] = frac.astype("float32")
    d["age_unit"] = a.cod_edad.astype("int8")
    d["age_group"] = pd.Categorical(
        pd.cut(d.age, AGE_BINS, labels=AGE_LABELS), categories=AGE_LABELS)
    d["pediatric"] = (d.age <= 19).astype("int8")
    d["sex"] = pd.Categorical(np.where(a.sexo == 1, "Male", "Female"), ["Male", "Female"])
    d["ethnicity"] = pd.Categorical(a.etnia.map(ETH_MAP), ETH_LEVELS)
    d["area"] = pd.Categorical(np.where(a.area_res == 1, "Urban", "Rural"), ["Urban", "Rural"])
    d["sector"] = pd.Categorical(np.where(a.sector == 1, "Public", "Private"),
                                 ["Public", "Private"])
    icd4 = a.cau_cie10.str.replace(".", "", regex=False).str[:4]
    d["ari_type"] = pd.Categorical(ari_type(a.causa3, icd4), TYPE_LEVELS)
    d["year"] = a.anio_ingr.astype("int16")

    # ---- wider admission-time feature set (new model) -------------------------------------
    for c in ["etnia", "nac_pac", "area_ubi", "clase", "tipo", "entidad", "mes_ingr",
              "dia_ingr", "esp_egrpa", "dia_estad", "anio_egr", "mes_egr"]:
        d[c] = a[c].astype("float32")
    d["sector3"] = a.sector.astype("float32")
    d["area_res"] = a.area_res.astype("float32")
    d["sexo"] = a.sexo.astype("float32")
    for c in ["prov_ubi", "cant_ubi", "prov_res", "cant_res"]:
        d[c] = a[c].astype("string")
    d["icd4"] = a.cau_cie10.str.replace(".", "", regex=False).str[:4]
    d["icd3"] = a.causa3.str[:3]
    d["file_year"] = a.file_year.astype("int16")
    date = pd.to_datetime(dict(year=a.anio_ingr, month=a.mes_ingr, day=a.dia_ingr),
                          errors="coerce")
    d["adm_date"] = date
    d["adm_dow"] = date.dt.dayofweek.astype("float32")
    d["adm_ym"] = (a.anio_ingr * 12 + a.mes_ingr - 1).astype("int32")
    d["same_prov"] = (a.prov_ubi == a.prov_res).astype("int8")
    d["same_cant"] = (a.cant_ubi == a.cant_res).astype("int8")
    return d


def sample(d, which):
    if which == "total":
        return d
    if which == "pediatric":
        return d[d.pediatric == 1]
    if which == "adult":
        return d[d.pediatric == 0]
    raise ValueError(which)


if __name__ == "__main__":
    d = load()
    print(len(d), int(d.dead.sum()), d.pediatric.sum(), d.isna().sum()[lambda s: s > 0].to_dict())
