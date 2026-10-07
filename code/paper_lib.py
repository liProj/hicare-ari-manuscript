"""Helpers shared by write_paper.py and write_report.py: number formatting, LaTeX tables, and
one place where every number quoted in the text is read from the result files.

Text templates use <<key>> placeholders (no f-strings around LaTeX, so no brace doubling); an
unresolved placeholder is a hard error.  Percentages go through pct(), which always emits an
escaped percent sign -- a bare % would silently comment out the rest of the line.
"""
import os
import re

import numpy as np
import pandas as pd

import cohort as C
import metrics as MT

JD = C.JD
TAB = f"{JD}/results/tables"
SAMPLES = ["total", "pediatric", "adult"]
SNAME = {"total": "All ages", "pediatric": "Paediatric", "adult": "Adult"}
PNAME = {"cv": "Cross-validation", "temporal": "Temporal (test 2022--2023)",
         "forward": "Forward chaining (2018--2023)", "geo": "Geographic (unseen provinces)"}
MAIN = {"cv": "care", "geo": "care-portable", "temporal": "care-roll", "forward": "care-roll"}
REF = {"cv": "paper_lr", "temporal": "paper_lr_best", "forward": "paper_lr_best",
       "geo": "paper_lr"}
MLAB = {"auroc": "AUROC", "auprc": "AUPRC", "brier": "Brier score", "ipa": "Scaled Brier (IPA)",
        "logloss": "Log loss", "cal_int_abs": "$|$calibration intercept$|$",
        "cal_slope_err": "$|$calibration slope $-$ 1$|$", "ece": "ECE", "ici": "ICI",
        "sens_at_spec90": "Sensitivity at 90\\% specificity", "nb_low": "Net benefit, low threshold",
        "nb_mid": "Net benefit, mid threshold", "nb_high": "Net benefit, high threshold"}
MODEL = {
    "paper_lr": "Reference LR (7 variables)", "paper_lr_best": "Reference LR, best of six",
    "paper_lr_noyear": "Reference LR, year removed", "paper_lr_carry": "Reference LR, latest year carried",
    "paper_lr-roll": "Reference LR + rolling recal.", "paper_lr_noyear-roll": "Reference LR, year removed + rolling recal.",
    "paper_lr_carry-roll": "Reference LR, latest year + rolling recal.",
    "lgbm7": "LightGBM on the 7 reference variables", "lr_full": "Ridge LR, wide features",
    "lgbm": "LightGBM", "xgb": "XGBoost", "tabm": "TabM", "tabicl": "TabICLv2 (30k context)",
    "care": "CARE", "care-raw": "CARE, no recalibration", "care-roll": "CARE + rolling recal.",
    "care-logit": "CARE, log-odds averaging",
    "care-roll-cold": "CARE + rolling recal., cold start", "care-portable": "CARE, location-free",
    "care-stacked": "CARE, cross-fitted stacking", "care-stratum": "CARE, stratum-specific training",
    "care-no-lgbm": "CARE without LightGBM", "care-no-xgb": "CARE without XGBoost",
    "care-no-tabm": "CARE without TabM", "lgbm-cal": "LightGBM, recalibrated",
    "xgb-cal": "XGBoost, recalibrated", "tabm-cal": "TabM, recalibrated",
    "lgbm-roll": "LightGBM + rolling recal.", "xgb-roll": "XGBoost + rolling recal.",
    "tabm-roll": "TabM + rolling recal.",
    "student_hard": "CARE-Score, observed labels", "student_soft": "CARE-Score, distilled",
    "student_mix": "CARE-Score, mixed labels",
    "lgbm@no_context": "LightGBM, no context", "lgbm@plus_year": "LightGBM, context + year",
    "lgbm@year_noctx": "LightGBM, year instead of context", "lgbm@no_geo": "LightGBM, location-free",
    "lgbm@plus_post": "LightGBM + length of stay and discharge specialty"}


def esc(s):
    return (str(s).replace("\\", "\\textbackslash ").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("≥", "$\\geq$").replace("≤", "$\\leq$")
            .replace("–", "--").replace("−", "$-$").replace("×", "$\\times$"))


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def num(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


def big(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


def signed(x, d=3):
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.{d}f}"


def latex_table(rows, header, caption, label, align=None, size="\\footnotesize", note=None,
                wide=False):
    """rows: list of lists of already-formatted LaTeX strings ('---' = midrule)."""
    align = align or ("l" + "r" * (len(header) - 1))
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[htbp]", "\\centering", size,
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "---":
            out.append("\\midrule")
        elif isinstance(r, str):
            out.append(r)
        else:
            out.append(" & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\par\\vspace{{2pt}}\\parbox{{0.97\\linewidth}}{{\\scriptsize {note}}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


def fill(template, facts, draft=False):
    """Replace <<key>> placeholders.  A missing key is a hard error, except in a draft build,
    where it prints as [pending] so the manuscript can be proofread while runs finish."""
    def rep(m):
        k = m.group(1)
        if k not in facts:
            if draft:
                return "[pending]"
            raise KeyError(f"unresolved placeholder <<{k}>>")
        return str(facts[k])
    out = re.sub(r"<<([A-Za-z0-9_.\-@]+)>>", rep, template)
    return out


class Results:
    def __init__(self):
        self.S = pd.read_csv(f"{TAB}/metrics_summary.csv")
        self.L = pd.read_csv(f"{TAB}/metrics_long.csv")
        self.W = pd.read_csv(f"{TAB}/winloss.csv")
        bp = f"{TAB}/bootstrap_vs_paper.csv"
        self.B = pd.read_csv(bp) if os.path.exists(bp) else pd.DataFrame()
        self.choice = pd.read_csv(f"{TAB}/temporal_reference_choice.csv")
        self._idx = self.S.set_index(["protocol", "sample", "model", "metric"])

    def m(self, protocol, sample, model, metric, what="mean"):
        try:
            return float(self._idx.loc[(protocol, sample, model, metric), what])
        except KeyError:
            return float("nan")

    def has(self, protocol, sample, model):
        return np.isfinite(self.m(protocol, sample, model, "auroc"))

    def ms(self, protocol, sample, model, metric, d=3):
        """mean ± SD over seeds (SD omitted when a single seed is available)."""
        mu, sd, n = (self.m(protocol, sample, model, metric, w) for w in ("mean", "std", "count"))
        if not np.isfinite(mu):
            return "--"
        if n >= 2 and np.isfinite(sd):
            return f"{mu:.{d}f} $\\pm$ {sd:.{d}f}"
        return f"{mu:.{d}f}"

    def wins(self, protocol=None, sample=None):
        w = self.W
        if protocol:
            w = w[w.protocol == protocol]
        if sample:
            w = w[w["sample"] == sample]
        return int(w.care_wins.sum()), len(w)

    def lost(self, protocol, sample):
        w = self.W[(self.W.protocol == protocol) & (self.W["sample"] == sample) & ~self.W.care_wins]
        return list(w.metric)

    def best_ref(self, protocol, sample):
        c = self.choice[(self.choice.protocol == protocol) & (self.choice["sample"] == sample)]
        return c.best_reference.iloc[0] if len(c) else "paper_lr"

    def boot(self, protocol, sample, metric):
        if not len(self.B):
            return None
        b = self.B[(self.B.protocol == protocol) & (self.B["sample"] == sample) & (self.B.metric == metric)]
        return b.iloc[0] if len(b) else None
