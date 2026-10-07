"""All figures of the paper.  Each figure reads only result files written by the pipeline.

Usage: make_figures.py [fig01 fig07 ...]   (no argument = every figure)
A figure whose inputs are missing is reported and skipped; `--strict` turns that into an error.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import precision_recall_curve, roc_curve

import cohort as C
import features as F
import metrics as MT
from figstyle import CAT, COLOR, GRID, INK, INK2, MARKER, MUTED, SEQ, clean, plt, save

JD = F.JD
FIG = f"{JD}/figures"
TAB = f"{JD}/results/tables"
SAMPLES = ["total", "pediatric", "adult"]
SNAME = {"total": "All ages", "pediatric": "Paediatric (0–19 y)", "adult": "Adult (≥20 y)"}
MODEL_LABEL = {
    "paper_lr": "Reference LR (7 variables)", "paper_lr_best": "Reference LR (best year handling)",
    "paper_lr_noyear": "Reference LR, year removed", "paper_lr_carry": "Reference LR, latest year carried",
    "lgbm7": "LightGBM, 7 reference variables", "lr_full": "Ridge LR, wide features",
    "lgbm": "LightGBM", "xgb": "XGBoost", "tabm": "TabM", "tabicl": "TabICLv2 (30k context)",
    "care": "CARE (ours)", "care-raw": "CARE without recalibration", "care-logit": "CARE, log-odds averaging",
    "care-roll": "CARE + rolling recalibration", "care-portable": "CARE, location-free",
    "care-stacked": "CARE, fitted stacking weights", "care-stratum": "CARE, stratum-specific training",
    "care-no-lgbm": "CARE without LightGBM", "care-no-xgb": "CARE without XGBoost",
    "care-no-tabm": "CARE without TabM", "lgbm-cal": "LightGBM, recalibrated",
    "xgb-cal": "XGBoost, recalibrated", "tabm-cal": "TabM, recalibrated",
    "student_hard": "CARE-Score, observed labels", "student_soft": "CARE-Score, distilled",
    "student_mix": "CARE-Score, mixed labels"}
MCOL = {"paper_lr": COLOR["ref"], "paper_lr_best": COLOR["ref"], "paper_lr_noyear": COLOR["ref"],
        "paper_lr_carry": COLOR["ref"], "care": COLOR["care"], "lgbm": COLOR["lgbm"],
        "xgb": COLOR["xgb"], "tabm": COLOR["tabm"], "tabicl": COLOR["tabicl"],
        "student_soft": COLOR["student"], "student_hard": COLOR["student"],
        "student_mix": COLOR["student"]}

_f = None


def feats():
    global _f
    if _f is None:
        _f = F.build()
    return _f


def pred(protocol, sample, name, seed=0):
    for root in ("ensemble", "preds"):
        fp = f"{JD}/results/{root}/{protocol}/{sample}/{name}_s{seed}.parquet"
        if os.path.exists(fp):
            return pd.read_parquet(fp).set_index("idx").sort_index()
    raise FileNotFoundError(f"{protocol}/{sample}/{name}_s{seed}")


def yp(protocol, sample, name, seed=0):
    d = pred(protocol, sample, name, seed)
    return feats().dead.values[d.index.values].astype(float), d.p.values, d


def summary():
    return pd.read_csv(f"{TAB}/metrics_summary.csv")


def metric(S, protocol, sample, model, m):
    r = S[(S.protocol == protocol) & (S["sample"] == sample) & (S.model == model) & (S.metric == m)]
    return float(r["mean"].iloc[0]) if len(r) else np.nan


# ------------------------------------------------------------------------------------------
def fig01():
    """Cohort derivation."""
    raw = pd.read_parquet(C.COHORT)
    d = C.load()
    n_all = sum(len(pd.read_parquet(f"{JD}/data/cohort/all_{y}.parquet", columns=["file_year"]))
                for y in C.YEARS)
    n_pre = int((~raw.anio_ingr.between(2015, 2023)).sum())
    n_cov = len(raw) - n_pre - len(d)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f6f5f2", bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=GRID, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.8, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))

    box(0.14, 0.86, 0.72, 0.11, f"INEC hospital discharge registry, nine yearly files 2015–2023\n"
                               f"{n_all:,} discharges")
    box(0.14, 0.66, 0.72, 0.11, f"Principal diagnosis J00–J06, J09–J18, J20–J22, J85–J86, U07.1–U07.2\n"
                               f"{len(raw):,} hospitalisations")
    box(0.14, 0.42, 0.72, 0.11, f"Study cohort (identical to the reference article)\n"
                               f"{len(d):,} hospitalisations · {int(d.dead.sum()):,} in-hospital deaths",
        fc=SEQ[0], bold=True)
    box(0.93, 0.60, 0.44, 0.1, f"Admitted before 2015\n−{n_pre}", fc="#ffffff")
    box(0.93, 0.47, 0.44, 0.1, f"COVID-19 code, admitted before 2020\n−{n_cov} (rule not stated in the article)",
        fc="#ffffff")
    ped, adu = d[d.pediatric == 1], d[d.pediatric == 0]
    box(0.02, 0.18, 0.44, 0.13, f"Paediatric, 0–19 y\n{len(ped):,} hospitalisations\n"
                                 f"{int(ped.dead.sum()):,} deaths ({ped.dead.mean():.2%})")
    box(0.54, 0.18, 0.44, 0.13, f"Adult, ≥20 y\n{len(adu):,} hospitalisations\n"
                                 f"{int(adu.dead.sum()):,} deaths ({adu.dead.mean():.2%})")
    box(0.02, 0.0, 0.96, 0.1, "Validation: 5-fold CV × 3 seeds · temporal (train 2015–2021, test 2022–2023)\n"
                               "forward chaining 2018–2023 · 6-fold grouped by hospital province", fc="#ffffff")
    arrow(0.5, 0.86, 0.5, 0.77)
    arrow(0.5, 0.66, 0.5, 0.53)
    arrow(0.5, 0.61, 0.93, 0.65)
    arrow(0.5, 0.58, 0.93, 0.52)
    arrow(0.4, 0.42, 0.24, 0.31)
    arrow(0.6, 0.42, 0.76, 0.31)
    arrow(0.24, 0.18, 0.24, 0.1)
    arrow(0.76, 0.18, 0.76, 0.1)
    ax.set_xlim(0, 1.39)
    ax.set_ylim(-0.02, 1)
    save(fig, f"{FIG}/fig01_cohort_flow")


def fig02():
    """Reconciliation of printed numbers with the public registry."""
    dc = pd.read_csv(f"{TAB}/data_check.csv")
    t2 = pd.read_csv(f"{TAB}/rates_table2.csv")
    lr = pd.read_csv(f"{TAB}/logit_reproduction.csv", keep_default_na=False)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    panels = [
        (dc.article.clip(lower=0.5), dc.registry.clip(lower=0.5),
         f"Table 1 counts\n{int((dc['diff'] == 0).sum())}/{len(dc)} cells identical"),
        (t2.article.clip(lower=0.5), t2.ours.clip(lower=0.5),
         f"Table 2 crude rates\n{int(t2.exact.sum())}/{len(t2)} identical at 2 dp"),
        (np.r_[lr.or_, lr.or_lo, lr.or_hi], np.r_[lr.r_or, lr.r_or_lo, lr.r_or_hi],
         "Tables 5–7 crude ORs\n(estimates and CI limits)"),
        (np.r_[lr.aor, lr.aor_lo, lr.aor_hi], np.r_[lr.r_aor, lr.r_aor_lo, lr.r_aor_hi],
         "Tables 5–7 adjusted ORs\n(estimates and CI limits)")]
    for ax, (x, y, title) in zip(axes, panels):
        x, y = np.asarray(x, float), np.asarray(y, float)
        lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=1)
        ax.scatter(x, y, s=12, color=COLOR["care"], alpha=0.75, edgecolor="white", lw=0.3, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Printed in the article")
        clean(ax, "both")
    axes[0].set_ylabel("Recomputed from the public registry")
    fig.tight_layout()
    save(fig, f"{FIG}/fig02_reconciliation")


def fig03():
    """Monthly admissions by infection group and monthly in-hospital death proportion."""
    d = C.load()
    ym = pd.PeriodIndex(year=d.year.astype(int), month=d.mes_ingr.astype(int), freq="M").to_timestamp()
    t = d.ari_type.astype(str)
    grp = np.where(t == "COVID-19", "COVID-19", np.where(t.str.contains("pneumonia|Pneumonia"), "Pneumonia",
                   np.where(t.str.contains("upper"), "Upper respiratory", "Other lower respiratory")))
    order = ["Upper respiratory", "Other lower respiratory", "Pneumonia", "COVID-19"]
    cols = [CAT[2], CAT[3], CAT[0], CAT[1]]
    cnt = pd.crosstab(ym, grp)[order]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 1, 1]))
    axes[0].stackplot(cnt.index, cnt.T.values / 1000, colors=cols, labels=order, linewidth=0.6,
                      edgecolor="white")
    axes[0].set_ylabel("Admissions per month (thousands)")
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].set_title("ARI hospitalisations by month of admission and infection group")
    for ax, flag, name, col in ((axes[1], 0, "Adults (≥20 y)", CAT[0]), (axes[2], 1, "Children (0–19 y)", CAT[6])):
        s = d[d.pediatric == flag]
        r = s.groupby(ym[d.pediatric == flag]).dead.mean() * 100
        ax.plot(r.index, r.values, color=col)
        ax.set_ylabel("Deaths per 100 admissions")
        ax.set_title(f"In-hospital death proportion — {name}", fontsize=9)
        ax.set_ylim(0, None)
    for ax in axes:
        clean(ax)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=GRID, alpha=0.45, lw=0)
    axes[1].text(pd.Timestamp("2020-04-01"), axes[1].get_ylim()[1] * 0.9, "2020–2021", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, f"{FIG}/fig03_monthly_series")


def fig04():
    """Age-adjusted rates: printed vs recomputed, with the count-weighted joinpoint fit."""
    import reproduce_rates as R
    t4 = pd.read_csv(f"{TAB}/rates_table4.csv")
    J = pd.read_csv(f"{TAB}/joinpoint.csv")
    names = [("hosp_total", "Hospitalisation, all ages"), ("hosp_ped", "Hospitalisation, paediatric"),
             ("hosp_adult", "Hospitalisation, adult"), ("mort_total", "In-hospital mortality, all ages"),
             ("mort_ped", "In-hospital mortality, paediatric"), ("mort_adult", "In-hospital mortality, adult")]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for ax, (key, title) in zip(axes.ravel(), names):
        g = t4[t4.series == key].sort_values("year")
        ax.plot(g.year, g.article, "^", color=COLOR["ref"], label="Printed in the article", ms=6, zorder=3)
        ax.plot(g.year, g.ours, "o", color=COLOR["care"], mfc="white", mew=1.4,
                label="Recomputed (this study)", ms=6, zorder=4)
        jw = J[(J.series == key) & (J.rates == "article rates, count-weighted")]
        ju = J[(J.series == key) & (J.rates == "article rates, unweighted")]
        for jj, ls, lab, col in ((jw, "-", "Joinpoint fit, count-weighted", INK2),
                                 (ju, ":", "Joinpoint fit, unweighted", MUTED)):
            # redraw the fitted segments from the article's rates
            wts = None
            if "count" in lab:
                d = C.load()
                m = (d.dead == 1) if key.startswith("mort") else np.ones(len(d), bool)
                if key.endswith("ped"):
                    m = m & (d.pediatric == 1)
                if key.endswith("adult"):
                    m = m & (d.pediatric == 0)
                wts = d[m].groupby("year").size().reindex(C.YEARS).values
            k = len(jj) - 1
            fits = R.joinpoint(C.YEARS, g.article.values, wts)
            ft = [x for x in fits if x["k"] == k][0]
            x = np.array(C.YEARS, float)
            ly = np.log(g.article.values)
            jp = [s["end"] for s in ft["segments"][:-1]]
            w = np.ones(9) if wts is None else np.asarray(wts, float)
            _, beta, _ = R._fit_segments(x, ly, jp, w)
            Xd = np.column_stack([np.ones_like(x), x] + [np.clip(x - j, 0, None) for j in jp])
            ax.plot(x, np.exp(Xd @ beta), ls, color=col, lw=1.5, label=lab, zorder=2)
        txt = ";  ".join(f"{r.seg}: {r.our_apc:+.1f}% (printed {r.article_apc:+.1f}%)" for r in jw.itertuples())
        ax.set_title(f"{title}\nAPC {txt}", fontsize=7.6)
        ax.set_yscale("log")
        clean(ax)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Age-adjusted rate per 100,000 (log scale)")
    fig.tight_layout()
    save(fig, f"{FIG}/fig04_joinpoint")


def fig05():
    """Forest plot: printed vs re-estimated adjusted odds ratios, total sample."""
    lr = pd.read_csv(f"{TAB}/logit_reproduction.csv", keep_default_na=False)
    g = lr[lr["sample"] == "total"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ypos = np.arange(len(g))[::-1]
    ax.axvline(1, color=MUTED, lw=1)
    ax.errorbar(g.aor, ypos + 0.17, xerr=[g.aor - g.aor_lo, g.aor_hi - g.aor], fmt="^", color=COLOR["ref"],
                ms=5, lw=1.2, label="Printed in the article (Table 5)")
    ax.errorbar(g.r_aor, ypos - 0.17, xerr=[g.r_aor - g.r_aor_lo, g.r_aor_hi - g.r_aor], fmt="o",
                color=COLOR["care"], ms=4.5, lw=1.2, label="Re-estimated from the public registry")
    short = {"Unspecified acute lower respiratory infection": "Unspecified acute LRTI",
             "Suppurative and necrotic conditions": "Suppurative/necrotic LRT"}
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{short.get(l, l)}" for l in g.level], fontsize=7.5)
    prev = None
    for yv, v in zip(ypos, g.variable):
        if v != prev:
            ax.axhline(yv + 0.5, color=GRID, lw=0.8)
            ax.text(0.012, yv + 0.33, {"age_group": "Age group (ref. 20–29)", "sex": "Sex (ref. male)",
                                       "ethnicity": "Ethnicity (ref. Mestizo)", "area": "Residence (ref. urban)",
                                       "sector": "Sector (ref. public)", "ari_type": "Infection type (ref. upper ARI)",
                                       "year": "Admission year (ref. 2015)"}[v],
                    transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED, va="center")
            prev = v
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted odds ratio for in-hospital death (log scale, 95% CI)")
    ax.legend(loc="lower right")
    clean(ax, "x")
    fig.tight_layout()
    save(fig, f"{FIG}/fig05_or_forest")


def fig06():
    """Schematic of the CARE pipeline."""
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f6f5f2"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=GRID, lw=1))
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=6.7, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))

    box(0.0, 0.52, 0.23, 0.46, "Public discharge abstract",
        "age (hours → years), sex,\nethnicity (9), nationality\nICD-10 code (4 characters)\nresidence: province, canton, area\n"
        "facility: canton, class, type,\nentity, sector\ncare outside own province/canton\nadmission month, weekday")
    box(0.0, 0.02, 0.23, 0.42, "Lagged epidemic context",
        "previous 3 calendar months:\nARI volume, COVID-19 share,\ndeath proportion — national and\nhospital province; facility load\n(replaces the year dummy)",
        fc=SEQ[0])
    for i, (name, body, col) in enumerate([("LightGBM", "gradient-boosted trees,\nnative categoricals", COLOR["lgbm"]),
                                           ("XGBoost", "histogram trees on GPU,\ncategorical splits", COLOR["xgb"]),
                                           ("TabM", "parameter-efficient MLP\nensemble (k = 16), PLE", COLOR["tabm"])]):
        y = 0.70 - i * 0.32
        box(0.31, y, 0.21, 0.26, name, body, fc="#ffffff")
        ax.add_patch(plt.Rectangle((0.31, y), 0.012, 0.26, fc=col, ec="none"))
        arrow(0.23, 0.75, 0.31, y + 0.13)
        arrow(0.23, 0.23, 0.31, y + 0.13)
        arrow(0.52, y + 0.13, 0.60, 0.51)
    box(0.60, 0.36, 0.17, 0.30, "Equal-weight\nlogit mean", "\n\nno fitted weights", fc="#ffffff")
    box(0.82, 0.36, 0.18, 0.30, "Recalibration", "2-parameter logistic map\nfitted on inner-validation\nrecords only", fc=SEQ[0])
    arrow(0.77, 0.51, 0.82, 0.51)
    box(0.60, 0.02, 0.40, 0.24, "Distilled student: CARE-Score",
        "one logistic equation\n6 patient variables + 3 context numbers\ntrained on teacher probabilities", fc="#ffffff")
    ax.add_patch(plt.Rectangle((0.60, 0.02), 0.012, 0.24, fc=COLOR["student"], ec="none"))
    arrow(0.69, 0.36, 0.69, 0.26)
    ax.text(0.91, 0.75, "CARE\nrisk of in-hospital death", ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLOR["care"])
    arrow(0.91, 0.66, 0.91, 0.74)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1)
    save(fig, f"{FIG}/fig06_care_schematic")


MAIN = {"cv": "care", "geo": "care-portable", "temporal": "care-roll", "forward": "care-roll"}


def _two(protocol, sample):
    """(reference, CARE) under a protocol: the reference is the article's model, or the best of
    its six variants when predicting forward in time; CARE is the configuration that carries
    the claim under that protocol."""
    ref = "paper_lr" if protocol in ("cv", "geo") else "paper_lr_best"
    return yp(protocol, sample, ref), yp(protocol, sample, MAIN[protocol])


def fig07():
    """ROC curves."""
    S = summary()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for ax, s in zip(axes, SAMPLES):
        (y, pr, _), (y2, pc, _) = _two("cv", s)
        for yy, p, key, lab in ((y, pr, "ref", "Reference LR"), (y2, pc, "care", "CARE")):
            fpr, tpr, _ = roc_curve(yy, p)
            step = max(1, len(fpr) // 3000)
            auc = metric(S, "cv", s, "paper_lr" if key == "ref" else "care", "auroc")
            ax.plot(fpr[::step], tpr[::step], color=COLOR[key], label=f"{lab}  AUROC {auc:.3f}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.set_title(SNAME[s]); ax.set_xlabel("1 − specificity")
        ax.legend(loc="lower right"); clean(ax, "both")
    axes[0].set_ylabel("Sensitivity")
    fig.tight_layout()
    save(fig, f"{FIG}/fig07_roc")


def fig08():
    """Precision-recall curves."""
    S = summary()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for ax, s in zip(axes, SAMPLES):
        (y, pr, _), (y2, pc, _) = _two("cv", s)
        for yy, p, key, lab in ((y, pr, "ref", "Reference LR"), (y2, pc, "care", "CARE")):
            prec, rec, _ = precision_recall_curve(yy, p)
            step = max(1, len(rec) // 3000)
            ap = metric(S, "cv", s, "paper_lr" if key == "ref" else "care", "auprc")
            ax.plot(rec[::step], prec[::step], color=COLOR[key], label=f"{lab}  AUPRC {ap:.3f}")
        ax.axhline(y.mean(), color=MUTED, lw=1, ls="--")
        ax.text(0.98, y.mean(), " prevalence", color=MUTED, fontsize=7, va="bottom", ha="right")
        ax.set_title(SNAME[s]); ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylim(0, None)
        ax.legend(loc="upper right"); clean(ax, "both")
    axes[0].set_ylabel("Precision (positive predictive value)")
    fig.tight_layout()
    save(fig, f"{FIG}/fig08_pr")


def _calib_bins(y, p, bins=20):
    o = np.argsort(p, kind="stable")
    e = np.linspace(0, len(p), bins + 1).astype(int)
    return (np.array([p[o][e[i]:e[i + 1]].mean() for i in range(bins)]),
            np.array([y[o][e[i]:e[i + 1]].mean() for i in range(bins)]))


def fig09():
    """Calibration curves on log-log axes (20 quantile groups)."""
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    for row, protocol in enumerate(("cv", "temporal")):
        for ax, s in zip(axes[row], SAMPLES):
            (y, pr, _), (y2, pc, _) = _two(protocol, s)
            lo, hi = 1e9, 0
            for yy, p, key, lab in ((y, pr, "ref", "Reference LR"), (y2, pc, "care", "CARE")):
                x, o = _calib_bins(yy, p)
                ok = (x > 0) & (o > 0)
                ax.plot(x[ok], o[ok], marker=MARKER[key], color=COLOR[key], label=lab, ms=4.5, lw=1.4,
                        mec="white", mew=0.5)
                lo, hi = min(lo, x[ok].min(), o[ok].min()), max(hi, x[ok].max(), o[ok].max())
            ax.plot([lo * 0.7, hi * 1.4], [lo * 0.7, hi * 1.4], color=MUTED, lw=1, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{SNAME[s]} — {'5-fold CV' if protocol == 'cv' else 'temporal test 2022–2023'}",
                         fontsize=9)
            clean(ax, "both")
            if row == 1:
                ax.set_xlabel("Predicted risk of death")
        axes[row, 0].set_ylabel("Observed death proportion")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, f"{FIG}/fig09_calibration")


def fig10():
    """Decision curves."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for ax, s in zip(axes, SAMPLES):
        (y, pr, _), (y2, pc, _) = _two("cv", s)
        prev = y.mean()
        ts = np.geomspace(prev / 8, min(0.6, prev * 6), 60)
        for yy, p, key, lab in ((y, pr, "ref", "Reference LR"), (y2, pc, "care", "CARE")):
            ax.plot(ts, [MT.net_benefit(yy, p, t) for t in ts], color=COLOR[key], label=lab)
        ax.plot(ts, [prev - (1 - prev) * t / (1 - t) for t in ts], color=MUTED, lw=1.2, ls="--", label="Flag all")
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        ax.set_ylim(-prev * 0.05, prev * 1.02)
        ax.set_xscale("log")
        ax.set_title(SNAME[s]); ax.set_xlabel("Risk threshold (log scale)")
        clean(ax, "both")
    axes[0].set_ylabel("Net benefit")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, f"{FIG}/fig10_decision_curves")


def _dot_grid(models, metrics_, protocol, fname, title_fmt, figsize):
    S = summary()
    fig, axes = plt.subplots(len(SAMPLES), len(metrics_), figsize=figsize, sharey=True)
    for i, s in enumerate(SAMPLES):
        for j, m in enumerate(metrics_):
            ax = axes[i, j]
            vals = [metric(S, protocol, s, mod, m) for mod in models]
            ypos = np.arange(len(models))[::-1]
            fin = [v for v in vals if np.isfinite(v)]
            if not fin:
                continue
            base = min(fin) if MT.HIGHER.get(m, True) else max(fin)
            for yv, mod, v in zip(ypos, models, vals):
                if not np.isfinite(v):
                    continue
                col = MCOL.get(mod, COLOR["care"] if mod.startswith("care") else MUTED)
                ax.plot([base, v], [yv, yv], color=col, lw=1.6, alpha=0.5)
                ax.plot(v, yv, "o", color=col, ms=6, mec="white", mew=0.8)
                ax.text(v, yv + 0.28, f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}", fontsize=6.3, ha="center",
                        color=INK2)
            ax.set_yticks(ypos)
            ax.set_yticklabels([MODEL_LABEL.get(mod, mod) for mod in models], fontsize=7.5)
            ax.set_ylim(-0.6, len(models) - 0.3)
            if i == 0:
                ax.set_title(MT.LABEL.get(m, m) + (" ↑" if MT.HIGHER.get(m, True) else " ↓"), fontsize=9)
            if j == 0:
                ax.set_ylabel(SNAME[s], fontsize=8.5, color=INK)
            clean(ax, "x")
            ax.margins(x=0.12)
    fig.tight_layout()
    save(fig, f"{FIG}/{fname}")


def fig11():
    """All models, three samples, cross-validation."""
    _dot_grid(["paper_lr", "lgbm7", "lr_full", "tabicl", "lgbm", "xgb", "tabm", "care"],
              ["auroc", "auprc", "logloss", "ici"], "cv", "fig11_model_comparison", "", (11.5, 8.2))


def fig12():
    """Win / loss map across protocols, samples and metrics."""
    W = pd.read_csv(f"{TAB}/winloss.csv")
    B = pd.read_csv(f"{TAB}/bootstrap_vs_paper.csv") if os.path.exists(f"{TAB}/bootstrap_vs_paper.csv") else None
    cols = [(p, s) for p in ("cv", "temporal", "forward", "geo") for s in SAMPLES
            if len(W[(W.protocol == p) & (W["sample"] == s)])]
    mets = list(MT.HIGHER)
    fig, ax = plt.subplots(figsize=(1.5 + 0.82 * len(cols), 5.6))
    for j, (p, s) in enumerate(cols):
        for i, m in enumerate(mets):
            r = W[(W.protocol == p) & (W["sample"] == s) & (W.metric == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            win = bool(r.care_wins)
            sig = ""
            if B is not None:
                b = B[(B.protocol == p) & (B["sample"] == s) & (B.metric == m)]
                if len(b) and not (b.lo.iloc[0] <= 0 <= b.hi.iloc[0]):
                    sig = "*"
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94,
                                       fc=SEQ[1] if win else "#f7c9b6", ec="white", lw=1.5))
            if m in ("cal_int_abs", "cal_slope_err", "ece", "ici"):
                # a relative change against a reference value that is ~0 is meaningless:
                # print the two values instead (CARE | reference)
                txt = f"{'✓' if win else '✗'} {r.care:.3f} | {r.reference:.3f}{sig}"
            else:
                rel = (r.care - r.reference) / abs(r.reference) if r.reference else np.nan
                txt = f"{'✓' if win else '✗'} {rel:+.0%}{sig}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=6.3, color=INK)
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(mets), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    pn = {"cv": "CV", "temporal": "Temporal", "forward": "Forward", "geo": "Geographic"}
    sn = {"total": "all", "pediatric": "paed.", "adult": "adult"}
    ax.set_xticklabels([f"{pn[p]}\n{sn[s]}" for p, s in cols], fontsize=7.5)
    ax.set_yticks(np.arange(len(mets)) + 0.5)
    ax.set_yticklabels([MT.LABEL[m] for m in mets], fontsize=8)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    tot = int(W.care_wins.sum())
    ax.set_xlabel(f"CARE better (✓, blue) or worse (✗, orange) than the reference. Relative change of the metric; for the "
                  f"four calibration rows the two values (CARE | reference).\n* = 95% bootstrap CI of the difference "
                  f"excludes 0.   CARE better in {tot} of {len(W)} comparisons.", fontsize=7.5)
    fig.tight_layout()
    save(fig, f"{FIG}/fig12_winloss")


def fig13():
    """Forward-chaining validation by test year."""
    f = feats()
    names = [("paper_lr", "Reference LR, unseen year = 2015", COLOR["ref"], ":", "^"),
             ("paper_lr_carry", "Reference LR, latest year carried", COLOR["ref"], "--", "v"),
             ("paper_lr_noyear", "Reference LR, year removed", COLOR["ref"], "-", "s"),
             ("care", "CARE, static", COLOR["care"], ":", "o"),
             ("care-roll", "CARE + rolling recalibration", COLOR["care"], "-", "D")]
    mets = [("auroc", "AUROC"), ("ipa", "Scaled Brier score (IPA)"), ("oe", "Observed / expected deaths")]
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 8.0), sharex=True)
    for i, s in enumerate(SAMPLES):
        for name, lab, col, ls, mk in names:
            try:
                d = pred("forward", s, name)
            except FileNotFoundError:
                continue
            y = f.dead.values[d.index.values].astype(float)
            rows = []
            for yr in sorted(d.fold.unique()):
                m = d.fold.values == yr
                r = MT.all_metrics(y[m], d.p.values[m], s)
                rows.append((yr, r["auroc"], r["ipa"], r["oe"]))
            rows = np.array(rows)
            for j in range(3):
                axes[i, j].plot(rows[:, 0], rows[:, j + 1], ls=ls, marker=mk, color=col, label=lab, ms=5,
                                mec="white", mew=0.6, lw=1.6)
        for j, (_, lab) in enumerate(mets):
            ax = axes[i, j]
            clean(ax)
            if i == 0:
                ax.set_title(lab, fontsize=9)
            if j == 2:
                ax.axhline(1, color=MUTED, lw=1, ls="--")
                ax.set_yscale("log")
            if j == 1:
                ax.axhline(0, color=MUTED, lw=1, ls="--")
        axes[i, 0].set_ylabel(SNAME[s], color=INK)
    for ax in axes[2]:
        ax.set_xlabel("Test year (model trained on all earlier years)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, f"{FIG}/fig13_forward_validation")


def fig14():
    """Geographic validation: AUROC and O/E per held-out province."""
    f = feats()
    r_, c_ = pred("geo", "total", "paper_lr"), pred("geo", "total", "care-portable")
    assert r_.index.equals(c_.index)
    y = f.dead.values[c_.index.values].astype(float)
    prov = f.prov_ubi.values[c_.index.values]
    rows = []
    for pv in sorted(set(prov)):
        m = prov == pv
        if y[m].sum() < 30:
            continue
        a, b = MT.all_metrics(y[m], r_.p.values[m], "total"), MT.all_metrics(y[m], c_.p.values[m], "total")
        rows.append(dict(prov=C.PROVINCES[pv], n=int(m.sum()), ref_auc=a["auroc"], care_auc=b["auroc"],
                         ref_ll=a["logloss"], care_ll=b["logloss"]))
    G = pd.DataFrame(rows).sort_values("care_auc")
    G.to_csv(f"{TAB}/geo_by_province.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 6.0), sharey=True)
    for ax, (a, b, lab) in zip(axes, (("ref_auc", "care_auc", "AUROC in the held-out province ↑"),
                                      ("ref_ll", "care_ll", "Log loss in the held-out province ↓"))):
        ypos = np.arange(len(G))
        ax.hlines(ypos, G[a], G[b], color=GRID, lw=2)
        ax.plot(G[a], ypos, "^", color=COLOR["ref"], label="Reference LR", ms=6, mec="white", mew=0.6)
        ax.plot(G[b], ypos, "D", color=COLOR["care"], label="CARE, location-free", ms=5.5, mec="white", mew=0.6)
        ax.set_yticks(ypos); ax.set_yticklabels([f"{p}  (n={n:,})" for p, n in zip(G.prov, G.n)], fontsize=7.5)
        ax.set_xlabel(lab)
        clean(ax, "x")
    axes[0].legend(loc="lower right")
    win_a, win_l = int((G.care_auc > G.ref_auc).sum()), int((G.care_ll < G.ref_ll).sum())
    fig.suptitle(f"Models never trained on the province they are tested on — CARE better in {win_a}/{len(G)} "
                 f"provinces (AUROC), {win_l}/{len(G)} (log loss)", fontsize=9.5, y=0.995)
    fig.tight_layout()
    save(fig, f"{FIG}/fig14_geographic")


def fig15():
    """Feature-group ablation of the LightGBM member."""
    S = summary()
    groups = ["diagnosis", "demography", "facility", "context", "residence", "calendar", "referral"]
    full = {m: metric(S, "cv", "total", "lgbm", m) for m in ("auroc", "logloss", "auprc")}
    rows = []
    for g in groups:
        for s in SAMPLES:
            rows.append(dict(group=g, sample=s,
                             d_auroc=metric(S, "cv", s, f"lgbm@no_{g}", "auroc") - metric(S, "cv", s, "lgbm", "auroc"),
                             d_ll=metric(S, "cv", s, f"lgbm@no_{g}", "logloss") - metric(S, "cv", s, "lgbm", "logloss")))
    A = pd.DataFrame(rows)
    A.to_csv(f"{TAB}/ablation_feature_groups.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    w = 0.26
    cols = {"total": CAT[0], "pediatric": CAT[6], "adult": CAT[2]}
    order = A[A["sample"] == "total"].sort_values("d_ll", ascending=False).group.tolist()
    for ax, (col, lab) in zip(axes, (("d_auroc", "Change in AUROC when the group is removed"),
                                     ("d_ll", "Change in log loss when the group is removed"))):
        for k, s in enumerate(SAMPLES):
            v = [A[(A.group == g) & (A["sample"] == s)][col].iloc[0] for g in order]
            ax.barh(np.arange(len(order)) + (1 - k) * w, v, height=w * 0.9, color=cols[s], label=SNAME[s])
        ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([g.capitalize() for g in order])
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(lab)
        clean(ax, "x")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, f"{FIG}/fig15_feature_ablation")


def fig16():
    """Calendar year vs lagged context, in-distribution and out of time."""
    S = summary()
    variants = [("lgbm@year_noctx", "Year label, no context"), ("lgbm@no_context", "Neither"),
                ("lgbm", "Lagged context (CARE member)"), ("lgbm@plus_year", "Context + year label")]
    mets = [("auroc", "AUROC ↑"), ("logloss", "Log loss ↓"), ("ici", "ICI ↓")]
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 7.2))
    protos = [("cv", "5-fold CV (years mixed)"), ("temporal", "Temporal test 2022–2023"),
              ("forward", "Forward chaining 2018–2023")]
    for i, (p, pl) in enumerate(protos):
        for j, (m, ml) in enumerate(mets):
            ax = axes[i, j]
            v = [metric(S, p, "total", name, m) for name, _ in variants]
            cols = [COLOR["ref"], MUTED, COLOR["care"], CAT[6]]
            ax.bar(range(4), v, color=cols, width=0.62)
            fin = [x for x in v if np.isfinite(x)]
            if fin:
                pad = (max(fin) - min(fin)) * 0.6 + 1e-6
                ax.set_ylim(min(fin) - pad, max(fin) + pad)
            for k, x in enumerate(v):
                if np.isfinite(x):
                    ax.text(k, x, f"{x:.4f}", ha="center", va="bottom", fontsize=7, color=INK2)
            ax.set_xticks(range(4))
            ax.set_xticklabels([l.replace(", ", ",\n").replace(" (", "\n(").replace(" + ", " +\n") for _, l in variants],
                               fontsize=6.6)
            if i == 0:
                ax.set_title(ml, fontsize=9)
            if j == 0:
                ax.set_ylabel(pl, fontsize=8, color=INK)
            clean(ax)
    fig.tight_layout()
    save(fig, f"{FIG}/fig16_year_vs_context")


def fig17():
    """Ensemble ablation."""
    _dot_grid(["lgbm", "xgb", "tabm", "care-no-lgbm", "care-no-xgb", "care-no-tabm", "care-logit",
               "care-raw", "care-stacked", "care-stratum", "care"],
              ["auroc", "logloss", "cal_int_abs", "cal_slope_err"], "cv", "fig17_ensemble_ablation", "", (11.5, 10.5))


def _subgroups(f):
    t = f.ari_type.astype(str)
    return [("Age group", f.age_group.astype(str), C.AGE_LABELS), ("Sex", f.sex.astype(str), ["Male", "Female"]),
            ("Ethnicity", f.ethnicity.astype(str), C.ETH_LEVELS), ("Residence", f.area.astype(str), ["Urban", "Rural"]),
            ("Sector", f.sector.astype(str), ["Public", "Private"]),
            ("Infection type", t, C.TYPE_LEVELS), ("Admission year", f.year.astype(str), [str(y) for y in C.YEARS])]


def fig18():
    """Subgroup discrimination and calibration."""
    f = feats()
    r_, c_ = pred("cv", "total", "paper_lr"), pred("cv", "total", "care")
    assert r_.index.equals(c_.index)
    y = f.dead.values[c_.index.values].astype(float)
    sub = f.iloc[c_.index.values]
    rows = []
    for gname, col, levels in _subgroups(sub):
        for lv in levels:
            m = (col.values == lv)
            if y[m].sum() < 25 or (1 - y[m]).sum() < 25:
                continue
            rows.append(dict(group=gname, level=lv, n=int(m.sum()), deaths=int(y[m].sum()),
                             ref_auc=MT.fast_auc(y[m], r_.p.values[m]), care_auc=MT.fast_auc(y[m], c_.p.values[m]),
                             ref_oe=y[m].mean() / r_.p.values[m].mean(), care_oe=y[m].mean() / c_.p.values[m].mean(),
                             ref_ll=MT.logloss(y[m], r_.p.values[m]), care_ll=MT.logloss(y[m], c_.p.values[m])))
    G = pd.DataFrame(rows)
    G.to_csv(f"{TAB}/subgroups.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 9.4), sharey=True)
    ypos = np.arange(len(G))[::-1]
    short = {"Unspecified acute lower respiratory infection": "Unspecified acute LRTI",
             "Suppurative and necrotic conditions": "Suppurative/necrotic LRT",
             "Acute upper respiratory infections": "Acute upper ARI"}
    for ax, (a, b, lab) in zip(axes, (("ref_auc", "care_auc", "AUROC within the subgroup ↑"),
                                      ("ref_oe", "care_oe", "Observed / expected deaths (1 = calibrated)"))):
        ax.hlines(ypos, G[a], G[b], color=GRID, lw=2)
        ax.plot(G[a], ypos, "^", color=COLOR["ref"], label="Reference LR", ms=6, mec="white", mew=0.6)
        ax.plot(G[b], ypos, "D", color=COLOR["care"], label="CARE", ms=5.5, mec="white", mew=0.6)
        ax.set_xlabel(lab)
        clean(ax, "x")
    axes[1].axvline(1, color=MUTED, lw=1, ls="--")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([f"{short.get(l, l)}" for l in G.level], fontsize=7.5)
    prev = None
    for yv, g in zip(ypos, G.group):
        if g != prev:
            for ax in axes:
                ax.axhline(yv + 0.5, color=GRID, lw=0.8)
            axes[1].text(1.0, yv + 0.32, g, transform=axes[1].get_yaxis_transform(), fontsize=7, color=MUTED,
                         ha="right", va="center")
            prev = g
    axes[0].legend(loc="lower left")
    n_auc = int((G.care_auc > G.ref_auc).sum())
    n_oe = int((np.abs(np.log(G.care_oe)) < np.abs(np.log(G.ref_oe))).sum())
    fig.suptitle(f"CARE has the higher AUROC in {n_auc}/{len(G)} subgroups and the O/E ratio closer to 1 in "
                 f"{n_oe}/{len(G)}", fontsize=9.5, y=0.995)
    fig.tight_layout()
    save(fig, f"{FIG}/fig18_subgroups")


def fig19():
    """TreeSHAP attribution of the LightGBM member."""
    imp = pd.read_csv(f"{TAB}/shap_importance.csv")
    grp = pd.read_csv(f"{TAB}/shap_groups.csv")
    dep = pd.read_parquet(f"{TAB}/shap_age.parquet")
    nice = {"age_cont": "Age (continuous)", "icd4": "ICD-10 code (4 char.)", "fac": "Facility group",
            "entidad": "Managing entity", "clase": "Facility class", "cant_ubi": "Hospital canton",
            "ctx_nat_cfr_adult": "Context: adult death prop. (national)", "ctx_prov_cfr": "Context: death prop. (province)",
            "ctx_nat_covid": "Context: COVID-19 share (national)", "ctx_prov_covid": "Context: COVID-19 share (province)",
            "ctx_nat_cfr": "Context: death prop. (national)", "ctx_fac_n": "Context: facility volume",
            "ctx_fac_ratio": "Context: facility load ratio", "ctx_prov_n": "Context: province volume",
            "ctx_nat_n": "Context: national volume", "icd3": "ICD-10 category (3 char.)", "ari_type_c": "Infection type (10)",
            "sexo": "Sex", "etnia": "Ethnicity (9)", "cant_res": "Canton of residence", "prov_res": "Province of residence",
            "prov_ubi": "Hospital province", "sector3": "Sector (3)", "tipo": "Facility type", "mes_ingr": "Admission month",
            "adm_dow": "Admission weekday", "dia_ingr": "Admission day of month", "same_prov": "Treated in own province",
            "same_cant": "Treated in own canton", "area_res": "Residence area", "area_ubi": "Hospital area",
            "nac_pac": "Nationality", "age_unit": "Age unit"}
    gcol = {"demography": CAT[0], "diagnosis": CAT[1], "facility": CAT[2], "context": CAT[3], "residence": CAT[4],
            "calendar": CAT[5], "referral": CAT[6]}
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[:, 0])
    top = imp.head(20).iloc[::-1]
    ax.barh([nice.get(f_, f_) for f_ in top.feature], top.mean_abs_shap, color=[gcol[g] for g in top.group], height=0.7)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("Top 20 features")
    clean(ax, "x")
    ax2 = fig.add_subplot(gs[0, 1])
    g2 = grp.sort_values("share")
    ax2.barh([g.capitalize() for g in g2.group], g2.share * 100, color=[gcol[g] for g in g2.group], height=0.7)
    for v, yv in zip(g2.share * 100, range(len(g2))):
        ax2.text(v + 0.5, yv, f"{v:.0f}%", va="center", fontsize=7.5, color=INK2)
    ax2.set_xlabel("Share of total mean |SHAP| (%)")
    ax2.set_title("By feature group")
    clean(ax2, "x")
    ax3 = fig.add_subplot(gs[1, 1])
    for k, (g, col) in enumerate((("Other ARI", CAT[2]), ("Pneumonia", CAT[0]), ("COVID-19", CAT[1]))):
        s = dep[dep.group == g]
        b = pd.cut(s.age, [-0.01, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 120])
        m = s.groupby(b, observed=True).agg(age=("age", "mean"), sh=("shap_age", "mean"))
        ax3.plot(m.age, m.sh, marker="o", ms=3.5, color=col, label=g, lw=1.6)
    ax3.axhline(0, color=MUTED, lw=1, ls="--")
    ax3.set_xlabel("Age (years)"); ax3.set_ylabel("SHAP of age (log-odds)")
    ax3.set_title("Age effect by infection group")
    ax3.legend(fontsize=7)
    clean(ax3)
    fig.tight_layout()
    save(fig, f"{FIG}/fig19_shap")


def fig20():
    """Distillation: student variants against the reference and the teacher."""
    S = summary()
    models = ["paper_lr", "paper_lr_noyear", "student_hard", "student_mix", "student_soft", "lgbm", "care"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)
    for i, (p, pl) in enumerate((("cv", "5-fold CV"), ("temporal", "Temporal test 2022–2023"))):
        for j, s in enumerate(SAMPLES):
            ax = axes[i, j]
            mods = [("paper_lr_best" if (p != "cv" and m == "paper_lr") else
                     "care-roll" if (p != "cv" and m == "care") else m) for m in models]
            vals = [metric(S, p, s, m, "logloss") for m in mods]
            ypos = np.arange(len(mods))[::-1]
            fin = [v for v in vals if np.isfinite(v)]
            if not fin:
                continue
            for yv, m, v in zip(ypos, mods, vals):
                if not np.isfinite(v):
                    continue
                col = MCOL.get(m, MUTED)
                ax.plot([max(fin), v], [yv, yv], color=col, lw=1.6, alpha=0.5)
                ax.plot(v, yv, "o", color=col, ms=6, mec="white", mew=0.8)
                ax.text(v, yv + 0.3, f"{v:.4f}  (AUROC {metric(S, p, s, m, 'auroc'):.3f})", fontsize=6.3,
                        ha="center", color=INK2)
            ax.set_yticks(ypos); ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in mods], fontsize=7.5)
            ax.set_ylim(-0.6, len(mods) - 0.2)
            ax.set_title(f"{SNAME[s]} — {pl}", fontsize=9)
            ax.margins(x=0.25)
            clean(ax, "x")
            if i == 1:
                ax.set_xlabel("Log loss ↓")
    fig.tight_layout()
    save(fig, f"{FIG}/fig20_distillation")


def fig21():
    """Risk concentration: observed mortality by predicted-risk decile and deaths captured."""
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2))
    rows = []
    for j, s in enumerate(SAMPLES):
        (y, pr, _), (y2, pc, _) = _two("cv", s)
        for yy, p, key, lab, off in ((y, pr, "ref", "Reference LR", -0.2), (y2, pc, "care", "CARE", 0.2)):
            o = np.argsort(-p, kind="stable")
            ys = yy[o]
            dec = np.array_split(ys, 10)
            rate = np.array([d.mean() for d in dec])[::-1] * 100
            axes[0, j].bar(np.arange(1, 11) + off, rate, width=0.38, color=COLOR[key], label=lab)
            cum = np.cumsum(ys) / ys.sum()
            x = np.arange(1, len(ys) + 1) / len(ys)
            step = max(1, len(x) // 2000)
            axes[1, j].plot(x[::step] * 100, cum[::step] * 100, color=COLOR[key], label=lab)
            for top in (0.05, 0.10, 0.20):
                rows.append(dict(sample=s, model=key, top=top, deaths_captured=float(cum[int(len(ys) * top) - 1])))
        axes[0, j].set_title(SNAME[s]); axes[0, j].set_xlabel("Decile of predicted risk (10 = highest)")
        axes[0, j].set_xticks(range(1, 11))
        axes[1, j].plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--")
        axes[1, j].set_xlabel("Hospitalisations flagged, highest risk first (%)")
        clean(axes[0, j]); clean(axes[1, j], "both")
    axes[0, 0].set_ylabel("Observed deaths per 100 admissions")
    axes[1, 0].set_ylabel("Deaths captured (%)")
    axes[0, 0].legend(loc="upper left"); axes[1, 0].legend(loc="lower right")
    pd.DataFrame(rows).to_csv(f"{TAB}/deaths_captured.csv", index=False)
    fig.tight_layout()
    save(fig, f"{FIG}/fig21_risk_concentration")


def fig22():
    """Observed and predicted risk across age, by infection group."""
    f = feats()
    r_, c_ = pred("cv", "total", "paper_lr"), pred("cv", "total", "care")
    sub = f.iloc[c_.index.values]
    y = sub.dead.values.astype(float)
    t = sub.ari_type.astype(str)
    grp = np.where(t == "COVID-19", "COVID-19", np.where(t.str.contains("pneumonia"), "Pneumonia", "Other ARI"))
    edges = np.r_[0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 120]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for ax, g in zip(axes, ("Other ARI", "Pneumonia", "COVID-19")):
        m = grp == g
        b = np.digitize(sub.age_cont.values[m], edges) - 1
        df = pd.DataFrame(dict(b=b, y=y[m], r=r_.p.values[m], c=c_.p.values[m], age=sub.age_cont.values[m]))
        a = df.groupby("b").agg(age=("age", "mean"), y=("y", "mean"), r=("r", "mean"), c=("c", "mean"), n=("y", "size"))
        a = a[a.n >= 200]
        ax.plot(a.age, a.y * 100, "o", color=INK, ms=4, label="Observed", zorder=4)
        ax.plot(a.age, a.r * 100, "-", color=COLOR["ref"], label="Reference LR", drawstyle="default")
        ax.plot(a.age, a.c * 100, "-", color=COLOR["care"], label="CARE")
        ax.set_title(g); ax.set_xlabel("Age (years)")
        ax.set_yscale("log")
        clean(ax, "both")
    axes[0].set_ylabel("Deaths per 100 admissions (log scale)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, f"{FIG}/fig22_age_risk")


def fig23():
    """Seed-to-seed stability of the headline metrics."""
    L = pd.read_csv(f"{TAB}/metrics_long.csv")
    mets = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("logloss", "Log loss"), ("ici", "ICI")]
    fig, axes = plt.subplots(3, 4, figsize=(10.6, 6.4))
    for i, s in enumerate(SAMPLES):
        for j, (m, ml) in enumerate(mets):
            ax = axes[i, j]
            for k, (mod, key) in enumerate((("paper_lr", "ref"), ("care", "care"))):
                v = L[(L.protocol == "cv") & (L["sample"] == s) & (L.model == mod) & (L.metric == m)].sort_values("seed")
                ax.plot(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v.value, MARKER[key], color=COLOR[key],
                        ms=6, mec="white", mew=0.7)
                if len(v):
                    ax.text(k, v.value.mean(), f"   {v.value.mean():.4f}\n   ±{v.value.std():.4f}", fontsize=6.5,
                            va="center", color=INK2)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Reference LR", "CARE"], fontsize=7.5)
            ax.set_xlim(-0.5, 1.9)
            if i == 0:
                ax.set_title(ml, fontsize=9)
            if j == 0:
                ax.set_ylabel(SNAME[s], color=INK, fontsize=8.5)
            clean(ax)
    fig.tight_layout()
    save(fig, f"{FIG}/fig23_seed_stability")


ALL = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12, fig13, fig14,
       fig15, fig16, fig17, fig18, fig19, fig20, fig21, fig22, fig23]


def main():
    strict = "--strict" in sys.argv
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    failed = []
    for fn in ALL:
        if want and fn.__name__ not in want:
            continue
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  SKIPPED {fn.__name__}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            plt.close("all")
    print("failed:", failed)
    if strict and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
