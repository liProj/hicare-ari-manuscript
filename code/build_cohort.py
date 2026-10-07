"""Build the acute-respiratory-infection (ARI) hospitalisation cohort from the INEC registry.

Reference article: Medicina 2026, 62, 1752 -- ARI hospitalisations and in-hospital mortality in
Ecuador, 2015-2023 (576,195 hospitalisations, 41,048 deaths).

INEC publishes one file per discharge year and changes the delivery format between years: some
years are coded CSV, others are CSV with value labels only (and no ICD-10 code at all in 2015,
2017 and 2021).  The SPSS release carries the numeric codes for every year, and the code tables
for the variables used here are stable across 2015-2023, so:

    2015, 2017, 2021, 2022, 2023   SPSS  (.sav, codes; value labels in the metadata)
    2016, 2018, 2019, 2020         coded CSV

Every year is reduced to the same coded columns and written to data/cohort/all_<year>.parquet
(all discharges, for denominators and context) and the ARI rows are concatenated into
data/cohort/ari.parquet.

Inclusion rule, exactly as printed in the article (section 2.1): three-character ICD-10 category
of the principal discharge diagnosis in J00-J06, J09-J18, J20-J22, J85-J86, or code U07.1/U07.2.
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyreadstat

JD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{JD}/data/inec"
SAV = f"{JD}/data/inec_spss"
OUT = f"{JD}/data/cohort"

KEEP = ["prov_ubi", "cant_ubi", "area_ubi", "clase", "tipo", "entidad", "sector", "mes_inv",
        "nac_pac", "sexo", "cod_edad", "edad", "etnia", "prov_res", "cant_res", "area_res",
        "anio_ingr", "mes_ingr", "dia_ingr", "anio_egr", "mes_egr", "dia_egr", "dia_estad",
        "con_egrpa", "esp_egrpa", "cau_cie10", "causa3"]
STR = ["prov_ubi", "cant_ubi", "prov_res", "cant_res", "cau_cie10", "causa3"]
SPSS_YEARS = (2015, 2017, 2021, 2022, 2023)


def _find(folder, ext):
    c = [p for p in glob.glob(f"{folder}/*{ext}") if "camas" not in os.path.basename(p).lower()]
    assert len(c) == 1, (folder, c)
    return c[0]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_year(year):
    if year in SPSS_YEARS:
        df, meta = pyreadstat.read_sav(_find(f"{SAV}/{year}", ".sav"), usecols=KEEP,
                                       apply_value_formats=False)
        n_file = meta.number_rows
    else:
        path = _find(f"{RAW}/{year}", ".csv")
        try:
            df = pd.read_csv(path, sep=";", dtype=str, usecols=KEEP, encoding="utf-8-sig")
        except UnicodeDecodeError:        # 2016 is Windows-1252; the kept columns are ASCII
            df = pd.read_csv(path, sep=";", dtype=str, usecols=KEEP, encoding="cp1252")
        n_file = len(df)
    assert len(df) == n_file
    out = pd.DataFrame(index=df.index)
    for c in KEEP:
        if c in STR:
            s = df[c].astype("string").str.strip().str.upper()
            # numeric-looking geography codes lose their leading zero in some releases
            if c in ("prov_ubi", "prov_res"):
                n = _num(s)
                s = n.round().astype("Int64").astype("string").str.zfill(2)
            elif c in ("cant_ubi", "cant_res"):
                n = _num(s)
                s = n.round().astype("Int64").astype("string").str.zfill(4)
            out[c] = s
        else:
            out[c] = _num(df[c]).astype("float32")
    out["file_year"] = np.int16(year)
    return out


def ari_mask(d):
    c3 = d.causa3.fillna("").str[:3]
    j = c3.str[0].eq("J")
    num = pd.to_numeric(c3.str[1:3], errors="coerce")
    in_j = j & (num.between(0, 6) | num.between(9, 18) | num.between(20, 22) | num.between(85, 86))
    c4 = d.cau_cie10.fillna("").str.replace(".", "", regex=False).str[:4]
    covid = c4.isin(["U071", "U072"])
    return in_j | covid


def work(year):
    d = load_year(year)
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(f"{OUT}/all_{year}.parquet", index=False)
    a = d[ari_mask(d)].copy()
    return year, len(d), a


def main():
    years = [int(a) for a in sys.argv[1:]] or list(range(2015, 2024))
    with ProcessPoolExecutor(min(9, len(years))) as ex:
        res = list(ex.map(work, years))
    parts = []
    for y, n, a in res:
        print(f"{y}: {n:,} discharges, {len(a):,} ARI, "
              f"{int((a.con_egrpa > 1).sum()):,} deaths", flush=True)
        parts.append(a)
    ari = pd.concat(parts, ignore_index=True)
    ari.to_parquet(f"{OUT}/ari.parquet", index=False)
    print(f"total ARI {len(ari):,}; deaths {int((ari.con_egrpa > 1).sum()):,}")


if __name__ == "__main__":
    main()
