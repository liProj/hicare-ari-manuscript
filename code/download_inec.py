"""Download the INEC hospital-discharge (egresos hospitalarios) microdata, 2015-2023.

Source: Registro Estadistico de Camas y Egresos Hospitalarios, INEC Ecuador, the open registry
named in the data availability statement of Medicina 2026, 62, 1752.  One archive per year goes
into its own directory under data/inec/<year>/.  An archive is accepted only after it has been
opened and every member CRC-checked -- a size test is not enough (a truncated download keeps a
plausible size and a valid header).
"""
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

JD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{JD}/data/inec"
ROOT = "https://www.ecuadorencifras.gob.ec/documentos"
OLD = f"{ROOT}/datos/investigaciones_sociales/Camas_Egresos_Hospitalarios/Egresos/Egresos%20spss"
NEW = f"{ROOT}/web-inec/Estadisticas_Sociales/Camas_Egresos_Hospitalarios"
URLS = {
    2015: f"{OLD}/bdd_egresos_hos_2015_spss-CSV.zip",
    2016: f"{OLD}/bdd_egresos_hos_2016_CSV.zip",
    2017: f"{OLD}/bdd_egresos_hos_2017_CSV.zip",
    2018: f"{NEW}/Cam_Egre_Hos_2018/camas%20egresos-csv.zip",
    2019: f"{NEW}/Cam_Egre_Hos_2019/BDD_Camas_y_Egresos_Hospitalarios_2019_CSV.zip",
    2020: f"{NEW}/Cam_Egre_Hos_2020/Datos_abiertos_Camas_Egresos_Hospitalarios_2020.zip",
    2021: f"{NEW}/Cam_Egre_Hos_2021/Datos_abiertos_camas_egresos_hospitalarios_2021.zip",
    2022: f"{NEW}/Cam_Egre_Hos_2022/Datos_Abiertos_camas_egresos_hospitalarios_2022.zip",
    2023: f"{NEW}/2023/datos_abiertos_camas_egresos_hospitalarios_2023.zip",
}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36"}


def valid(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and len(z.namelist()) > 0
    except Exception:
        return False


def get(year):
    d = f"{OUT}/{year}"
    os.makedirs(d, exist_ok=True)
    dst = f"{d}/egresos_{year}.zip"
    if os.path.exists(dst) and valid(dst):
        return year, "cached", os.path.getsize(dst)
    tmp = dst + ".part"
    for attempt in range(5):
        try:
            with requests.get(URLS[year], headers=UA, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
            if valid(tmp):
                os.replace(tmp, dst)
                if valid(dst):
                    return year, "ok", os.path.getsize(dst)
            print(year, "invalid archive, retrying", flush=True)
        except Exception as e:
            print(year, "error", repr(e)[:200], flush=True)
    return year, "FAILED", 0


def main():
    years = [int(a) for a in sys.argv[1:]] or sorted(URLS)
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(get, years))
    for y, s, n in res:
        print(f"{y} {s} {n/1e6:.1f} MB", flush=True)
    if any(s == "FAILED" for _, s, _ in res):
        sys.exit(1)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
