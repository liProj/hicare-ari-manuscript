"""Build Medicina2/screening.csv: an assessment of each of the 100 round-2 Medicina papers
(Crossref ISSN 1648-9144, published 2026-09-10 -> 2026-10-05, articles 1741-1923, none of them
in the round-1 exclusion list).

    score = 0.4*s_data + 0.3*s_time + 0.2*s_ease + 0.1*s_figs      s_figs = n_fig capped at 10

s_data is judged against what the data availability statement actually delivers -- every
repository link was resolved and every supplement opened -- not against how open the wording is.

  0 = review / no new data / explicitly restricted
  1 = data on request only
  2 = partial: summary tables printed or source gated behind credentialing
  3 = every result printed in the article as summary tables
  4 = complete machine-readable dataset released, downloaded, and its N reconciled
  5 = complete dataset AND the analysis code released

A paper can only be the round's target if, in addition, it has a measurable benchmark
(`benchmark` = 1): a fitted model whose out-of-sample performance a new method can be compared
with on enough data for the comparison to mean something.  The target is the highest-scoring
paper with benchmark = 1; papers that score higher but have no such benchmark are listed with
the reason they were passed over.
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.dirname(HERE)

VERIFIED = {
    1752: dict(
        data_availability="fully public and verified: INEC national hospital-discharge registry, "
                          "nine yearly microdata files 2015-2023 (CSV and SPSS)",
        data_location="https://www.ecuadorencifras.gob.ec/camas-y-egresos-hospitalarios/ "
                      "(+ year pages 2015-2023)",
        data_size="457 MB zipped / 4.3 GB CSV; 10,039,949 discharges, 576,195 in the cohort",
        method="joinpoint regression of age-adjusted ARI hospitalisation and in-hospital "
               "mortality rates; logistic regression of in-hospital death on 7 categorical "
               "variables in the total, paediatric and adult samples",
        reproducibility="excellent: all 210 cells of Table 1 reconcile exactly (576,195 "
                        "hospitalisations, 41,048 deaths) after three undocumented rules were "
                        "identified; 500/500 province-year rates, crude ORs and every joinpoint "
                        "APC reproduce; adjusted ORs reproduce to a median 2.4%",
        new_method_idea="context-aware risk ensemble (LightGBM + XGBoost + TabM) on the full "
                        "discharge abstract, lagged epidemic-context covariates instead of a "
                        "year dummy, leak-free recalibration, distilled logistic score",
        novelty="medium-high",
        can_beat="yes: the article's model is a fitted logistic regression on 41,048 events, so "
                 "discrimination, calibration and net benefit are directly comparable under "
                 "cross-validation, temporal and geographic validation",
        s_data=4, s_time=3, s_ease=3, benchmark=1,
        note="SELECTED"),
    1805: dict(
        data_availability="fully public and verified: OSF project with extraction data, "
                          "analysis-ready dataset and R code",
        data_location="https://osf.io/eyp5v/ (OSF_Reproducibility_Package-06_09_2026.zip)",
        data_size="199 kB zip; 15 trials / 1041 participants",
        method="random-effects meta-analysis (REML, Hartung-Knapp) of exercise on depressive "
               "symptoms in lung cancer",
        reproducibility="excellent on paper: data and code are both released",
        new_method_idea="Bayesian model-averaged or robust-variance meta-analysis",
        novelty="low",
        can_beat="no measurable benchmark: a pooled effect from 15 trials has no out-of-sample "
                 "performance to exceed, so 'better' could only mean a different estimate",
        s_data=5, s_time=5, s_ease=5, benchmark=0,
        note="highest formula score; passed over -- no performance benchmark"),
    1798: dict(
        data_availability="public and verified: de-identified dataset in the supplement",
        data_location="article supplement s001 (2.Supplementary_deidentified_data.xlsx)",
        data_size="xlsx, 83 rows x 18 columns -- matches the reported n = 83",
        method="multivariable linear regression of pharyngeal airway area on nasal "
               "cross-sectional area (adjusted R2 = 0.17)",
        reproducibility="good: N reconciles; a single small regression",
        new_method_idea="regularised / non-linear regression with repeated cross-validation",
        novelty="low",
        can_beat="not credibly: 83 patients and R2 = 0.17; any cross-validated gain would sit "
                 "inside the resampling noise",
        s_data=4, s_time=5, s_ease=5, benchmark=0,
        note="passed over -- sample too small for a meaningful comparison"),
    1757: dict(
        data_availability="aggregate series public and verified: annual case counts, "
                          "populations and age-standardised rates in the supplement; the "
                          "case-level registry is restricted",
        data_location="article supplement s001 (Supplementary_Table_S1_annual_data.csv)",
        data_size="csv, 22 annual rows (1999-2020) x 11 columns",
        method="joinpoint regression and ARIMA projection of oral-cavity cancer incidence",
        reproducibility="good at the aggregate level",
        new_method_idea="time-series foundation model or Bayesian structural forecast",
        novelty="low-medium",
        can_beat="not credibly: 22 annual points per series; a forecast to 2040 cannot be "
                 "scored and a rolling-origin test would have 3-5 evaluation points",
        s_data=4, s_time=5, s_ease=4, benchmark=0,
        note="passed over -- series too short to score a forecast"),
    1913: dict(
        data_availability="NOT freely public: MIMIC-IV v3.1 requires PhysioNet credentialing, "
                          "training and a data use agreement; supplement holds code and "
                          "aggregate tables only",
        data_location="https://doi.org/10.13026/kpb9-mt58 (gated)",
        data_size="1453 patients in the analysis cohort",
        method="restricted-cubic-spline logistic model of fluid-removal intensity and 90-day "
               "mortality with multiple imputation",
        reproducibility="blocked without credentialed access",
        new_method_idea="-", novelty="-", can_beat="n/a (gated data)",
        s_data=2, s_time=2, s_ease=2, benchmark=0, note="gated data"),
    1808: dict(
        data_availability="no dataset: the Zenodo record holds 12 demonstration videos only",
        data_location="https://doi.org/10.5281/zenodo.21318086",
        data_size="787 MB of .mov files",
        method="clinical protocol description and narrative synthesis",
        reproducibility="n/a (no data analysed)",
        new_method_idea="-", novelty="-", can_beat="n/a (no dataset)",
        s_data=0, s_time=1, s_ease=1, benchmark=0, note="videos only"),
    1814: dict(
        data_availability="public: study-level extraction, search logs and risk-of-bias sheets "
                          "in the supplement (systematic review, no primary data)",
        data_location="article supplement s001 (6 xlsx files)",
        data_size="study-level tables",
        method="systematic review of behavioural and psychosocial effects of CGM",
        reproducibility="good for the evidence tables; no quantitative model",
        new_method_idea="-", novelty="-", can_beat="n/a (narrative synthesis, no benchmark)",
        s_data=3, s_time=3, s_ease=3, benchmark=0, note="review"),
}

CLASS_DEFAULTS = {
    "NODATA/review": dict(
        data_availability="none (narrative or systematic review, no new data)",
        data_location="-", data_size="-", reproducibility="n/a (no primary data)",
        new_method_idea="-", novelty="-", can_beat="n/a (no dataset)",
        s_data=0, s_time=1, s_ease=1),
    "RESTRICTED": dict(
        data_availability="NOT public (restricted: patient privacy, ethics approval or "
                          "institutional policy)",
        data_location="-", data_size="unknown",
        reproducibility="blocked: the statement explicitly rules out release",
        new_method_idea="-", novelty="-", can_beat="n/a (data not public)",
        s_data=0, s_time=2, s_ease=2),
    "REQUEST": dict(
        data_availability="NOT public (available from the corresponding author on request)",
        data_location="-", data_size="unknown",
        reproducibility="blocked: no dataset accompanies the article",
        new_method_idea="-", novelty="-", can_beat="n/a (data not public)",
        s_data=1, s_time=2, s_ease=2),
    "ART": dict(
        data_availability="in article (results printed as summary tables)",
        data_location="article tables", data_size="group summaries only",
        reproducibility="easy at the summary level; individual patient records not released",
        new_method_idea="-", novelty="low",
        can_beat="no individual-level data, so no out-of-sample comparison is possible",
        s_data=3, s_time=4, s_ease=4),
    "ART+request": dict(
        data_availability="partly in article, remainder on request",
        data_location="article tables", data_size="group summaries only",
        reproducibility="partial", new_method_idea="-", novelty="low",
        can_beat="individual-level data withheld", s_data=2, s_time=3, s_ease=3),
    "PUBDB": dict(
        data_availability="derived from a public database, but the constructed analysis file "
                          "is not released",
        data_location="public database cited in the text", data_size="unknown",
        reproducibility="partial: the source is open, the derived cohort is not",
        new_method_idea="-", novelty="low",
        can_beat="reconstruction of the analysis cohort would be guesswork",
        s_data=2, s_time=2, s_ease=2),
}
# a DOI pointing at the article itself is not a data repository
SELF_DOI = re.compile(r"doi\.org/10\.3390/medicina", re.I)


def classify(row):
    a = int(row.article)
    if a in VERIFIED:
        return dict(VERIFIED[a])
    cat = str(row.das_cat)
    url = str(row.das_url or "")
    if cat.startswith("REPO") and (not url or SELF_DOI.search(url)):
        # the regex found only the article's own DOI, so fall back on the wording
        cat = "REQUEST" if "request" in cat.lower() else "ART"
    for key, val in CLASS_DEFAULTS.items():
        if cat.startswith(key):
            return dict(val)
    if cat.startswith("OTHER"):
        return dict(CLASS_DEFAULTS["NODATA/review"])
    return dict(CLASS_DEFAULTS["REQUEST"])


def main():
    raw = pd.read_csv(f"{JD}/screen_raw.csv")
    rows = []
    for _, r in raw.iterrows():
        rec = classify(r)
        rec.setdefault("method", str(r.title)[:160])
        rec.setdefault("benchmark", 0)
        rec.setdefault("note", "")
        n_fig = int(r.n_fig or 0)
        s_figs = min(n_fig, 10)
        score = (0.4 * rec["s_data"] + 0.3 * rec["s_time"] + 0.2 * rec["s_ease"]
                 + 0.1 * s_figs)
        rows.append(dict(article=int(r.article), doi=r.doi, published=r.published,
                         title=r.title, n_fig=n_fig, n_tab=int(r.n_tab or 0),
                         n_figS=int(r.n_figS or 0), n_tabS=int(r.n_tabS or 0),
                         das_cat=r.das_cat,
                         das_url=r.das_url if isinstance(r.das_url, str) else "",
                         supp_files=r.supp_files if isinstance(r.supp_files, str) else "",
                         **rec, s_figs=s_figs, score=round(score, 3)))
    S = pd.DataFrame(rows).sort_values(["score", "article"], ascending=[False, True])
    S["rank"] = range(1, len(S) + 1)
    S.to_csv(f"{JD}/screening.csv", index=False)
    print(f"{len(S)} papers -> {JD}/screening.csv")
    print(S[["rank", "article", "score", "s_data", "s_time", "s_ease", "n_fig", "benchmark",
             "note"]].head(12).to_string(index=False))
    print("\ns_data distribution:")
    print(S.s_data.value_counts().sort_index().to_string())
    print("target:", S[S.benchmark == 1].article.tolist())


if __name__ == "__main__":
    main()
