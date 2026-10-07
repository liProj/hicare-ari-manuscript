"""Classify the Data Availability Statement of each of the 100 round-2 Medicina papers
and print a per-category listing. Also pulls any dataset URL/DOI out of the DAS."""
import pandas as pd, re, os
JD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{JD}/screen_raw.csv")

URL = re.compile(r"(https?://[^\s,;)\]]+|doi\.org/10\.\d{4,9}/[^\s,;)\]]+|10\.5281/zenodo\.\d+)", re.I)

def cat(s):
    s = str(s).lower()
    if re.search(r"no new data (were|was)|not applicable|data sharing (is|does) not", s):
        return "NODATA/review"
    if re.search(r"unavailable due to privacy|not publicly available|ethical restriction|"
                 r"confidential|commercial(ly)? (sensitive|restriction)|proprietary|"
                 r"patient privacy|institutional review board|ethics committee approval is "
                 r"required|restrictions apply", s):
        return "RESTRICTED"
    if URL.search(s) and not re.search(r"on (reasonable )?request", s):
        return "REPO/URL"
    if URL.search(s):
        return "REPO+request"
    if re.search(r"on (reasonable )?request|from the corresponding author|available by the authors", s):
        if re.search(r"included (in|within) the|within (the|this) article|supplementary|presented in the", s):
            return "ART+request"
        return "REQUEST"
    if re.search(r"(included|contained|presented|available|reported) (in|within) (the|this) (article|paper|manuscript|study|text)|"
                 r"in the manuscript|within the article|in the figures and tables|data are contained", s):
        return "ART"
    if re.search(r"public(ly)? (available )?(database|repositor|dataset)|faostat|world bank|"
                 r"eurostat|fadn|copernicus|usda|earth ?engine", s):
        return "PUBDB"
    return "OTHER:" + s[:70]

d["das_cat"] = d.das.apply(cat)
d["das_url"] = d.das.apply(lambda s: ";".join(dict.fromkeys(URL.findall(str(s))))[:300])
d.to_csv(f"{JD}/screen_raw.csv", index=False)
print(d.das_cat.value_counts().to_string())
print()
order = ["REPO/URL", "REPO+request", "PUBDB", "ART", "ART+request", "OTHER:", "REQUEST",
         "RESTRICTED", "NODATA/review"]
for c in order:
    sub = d[d.das_cat.str.startswith(c)]
    if not len(sub): continue
    print(f"=== {c} ({len(sub)}) ===")
    for _, r in sub.sort_values("article").iterrows():
        kw = str(r.model_kw)[:60]
        print(f"  #{r.article} fig{r.n_fig:>3} tab{r.n_tab:>3} figS{r.n_figS:>2} tabS{r.n_tabS:>2} "
              f"| {str(r.title)[:66]}")
        if r.das_url: print(f"        URL: {r.das_url[:150]}")
