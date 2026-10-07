"""Extract screening signals from Medicina texts: DAS, supplementary statement, figure/table
counts, model/metric keywords, sample-size statements. Writes Medicina/screen_raw.csv."""
import os, re, glob, sys
import pandas as pd

JD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(f"{JD}/latest50_crossref.csv")

def fname(doi):
    tag = re.search(r"10\.3390/[a-z]+(\d+)$", doi).group(1)
    art = int(tag[-4:]); vol = int(tag[:-6]) if len(tag) > 6 else int(tag[0])
    return f"medicina-{vol:02d}-{art:05d}", art

def squeeze(s):
    return re.sub(r"\s+", " ", s or "").strip()

MODEL_KW = ["machine learning", "deep learning", "random forest", "gradient boosting", "xgboost",
            "lightgbm", "neural network", "convolutional", "transformer", "radiomics",
            "nomogram", "decision curve", "calibration curve", "roc", "auc", "c-statistic",
            "c-index", "sensitivity", "specificity", "ppv", "npv", "likelihood ratio",
            "logistic regression", "cox regression", "proportional hazards", "kaplan-meier",
            "log-rank", "propensity score", "multivariable", "univariable", "odds ratio",
            "hazard ratio", "risk ratio", "confidence interval", "anova", "chi-square",
            "mann-whitney", "kruskal-wallis", "spearman", "pearson", "bland-altman",
            "intraclass correlation", "cohen's kappa", "mixed model", "gee",
            "mediation analysis", "structural equation", "factor analysis", "rasch",
            "item response", "cronbach", "meta-analysis", "prisma", "forest plot",
            "bioinformatics", "differential expression", "gene set enrichment", "qpcr",
            "rna-seq", "mirna", "sequencing", "elisa", "immunohistochemistry",
            "bootstrap", "cross-valid", "internal validation", "external validation",
            "discrimination", "net reclassification", "brier score"]
REPO_KW = ["zenodo", "dryad", "figshare", "osf.io", "github", "gitlab", "mendeley data",
           "dataverse", "kaggle", "physionet", "mimic", "uk biobank", "nhanes", "seer",
           "tcga", "geo accession", "gse", "sra accession", "bioproject", "arrayexpress",
           "clinicaltrials.gov", "prospero", "dbgap", "ega", "immport", "openneuro",
           "data.mendeley", "doi.org/10.5281", "doi.org/10.6084", "supplementary data file"]

rows = []
for _, r in df.iterrows():
    n, art = fname(r.doi)
    tp = f"{JD}/texts/{n}.txt"
    if not os.path.exists(tp):
        rows.append(dict(article=art, doi=r.doi, name=n, missing=1)); continue
    t = open(tp, errors="ignore").read()
    low = t.lower()
    m = re.search(r"data\s+availability\s+statement[:\s]*(.{0,600})", t, re.I | re.S)
    das = squeeze(m.group(1)) if m else ""
    m2 = re.search(r"supplementary\s+materials?[:\s]*(.{0,400})", t, re.I | re.S)
    supp = squeeze(m2.group(1)) if m2 else ""
    figs = [int(x) for x in re.findall(r"\bFigure\s+(\d{1,2})\b", t)]
    tabs = [int(x) for x in re.findall(r"\bTable\s+(\d{1,2})\b", t)]
    local = sorted(os.path.basename(p) for p in glob.glob(f"{JD}/supp/{n}-s*")
                   if os.path.isfile(p))
    sizes = {os.path.basename(p): os.path.getsize(p) for p in glob.glob(f"{JD}/supp/{n}-s*")
             if os.path.isfile(p)}
    rows.append(dict(
        article=art, doi=r.doi, name=n, published=r.published, title=squeeze(r.title),
        is_review=int(bool(re.search(r"\breview\b|perspectives|advances, challenges",
                                     str(r.title), re.I))),
        n_fig=max(figs) if figs else 0, n_tab=max(tabs) if tabs else 0,
        n_figS=len(set(re.findall(r"\bFigure\s+S\d+", t))),
        n_tabS=len(set(re.findall(r"\bTable\s+S\d+", t))),
        model_kw=";".join(k for k in MODEL_KW if k in low),
        repo_kw=";".join(k for k in REPO_KW if k in low),
        supp_files=";".join(f"{k}({v//1024}KB)" for k, v in sorted(sizes.items())),
        das=das[:500], supp_stmt=supp[:300], chars=len(t), missing=0))

out = pd.DataFrame(rows).sort_values("article", ascending=False)
out.to_csv(f"{JD}/screen_raw.csv", index=False)
print(len(out), "rows ->", f"{JD}/screen_raw.csv")
print("with supp files:", (out.supp_files.fillna("") != "").sum(),
      "| reviews:", out.is_review.sum())
