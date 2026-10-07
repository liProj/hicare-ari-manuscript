"""Figures of the second manuscript (population-health analyses + HiCARE).

Usage: v2_figures.py [m1 m2 ... t1 ...] [--strict]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import cohort as C
import features as F
import metrics as MT
from figstyle import CAT, COLOR, GRID, INK, INK2, MUTED, SEQ, clean, plt, save

JD = F.JD
T = f"{JD}/results/v2/tables"
FIG = f"{JD}/figures_v2"
FLAGC = {"high": CAT[1], "low": CAT[0], "as expected": MUTED}


def ym_to_date(ym):
    return pd.to_datetime(dict(year=np.asarray(ym) // 12, month=np.asarray(ym) % 12 + 1, day=1))


def m1():
    """Risk-standardised mortality by province."""
    r = pd.read_csv(f"{T}/smr_prov_ubi_rich.csv").sort_values("smr_eb")
    c = pd.read_csv(f"{T}/smr_prov_ubi_crude.csv").set_index("name").smr.reindex(r.name)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    y = np.arange(len(r))
    ax.axvline(1, color=INK2, lw=1)
    ax.hlines(y, r.lo, r.hi, color=[FLAGC[f_] for f_ in r.flag], lw=2.2)
    ax.scatter(r.smr_eb, y, s=38, color=[FLAGC[f_] for f_ in r.flag], zorder=3, edgecolor="white", lw=0.6,
               label="Case-mix standardised (95% interval)")
    ax.scatter(c.values, y, s=26, marker="|", color=INK, zorder=4, label="Crude (unadjusted)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ({int(o):,} deaths)" for n, o in zip(r.name, r.O)], fontsize=7.6)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.6, 0.8, 1, 1.25, 1.6])
    ax.set_xticklabels(["0.5", "0.6", "0.8", "1", "1.25", "1.6"])
    ax.set_xlabel("Standardised mortality ratio, observed / expected in-hospital deaths (log scale)")
    ax.legend(loc="lower right", fontsize=7.5)
    hi, lo = int((r.flag == "high").sum()), int((r.flag == "low").sum())
    ax.set_title(f"{hi} provinces above and {lo} below the deaths expected from their case-mix", fontsize=9.5)
    clean(ax, "x")
    fig.tight_layout()
    save(fig, f"{FIG}/m1_province_smr")


def m2():
    """Funnel plot of facility groups."""
    r = pd.read_csv(f"{T}/smr_fac_rich.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    e = np.geomspace(max(r.E.min(), 1), r.E.max() * 1.1, 200)
    for q, ls in ((0.975, "--"), (0.999, ":")):
        ax.plot(e, stats.poisson.ppf(q, e) / e, ls, color=INK2, lw=1)
        ax.plot(e, stats.poisson.ppf(1 - q, e) / e, ls, color=INK2, lw=1)
    for fl in ("as expected", "low", "high"):
        s = r[r.flag == fl]
        ax.scatter(s.E, s.smr, s=14 + 10 * np.log10(s.n), color=FLAGC[fl], alpha=0.75, edgecolor="white", lw=0.4,
                   label=f"{fl} ({len(s)})")
    ax.axhline(1, color=INK2, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_ylim(0.04, 6)
    ax.set_xlabel("Expected deaths given case-mix (log scale)")
    ax.set_ylabel("Observed / expected deaths (log scale)")
    ax.legend(title="Shrunken estimate vs 1", fontsize=7.5, title_fontsize=7.5, loc="lower right")
    ax.set_title(f"{len(r)} facility groups with ≥300 ARI admissions; dashed 95%, dotted 99.8% Poisson limits", fontsize=9)
    clean(ax, "both")
    fig.tight_layout()
    save(fig, f"{FIG}/m2_facility_funnel")


def m3():
    """Provider type and facility class."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for ax, key, title in ((axes[0], "entidad_i", "Managing entity"), (axes[1], "clase_i", "Facility class")):
        r = pd.read_csv(f"{T}/smr_{key}_rich.csv").sort_values("smr_eb")
        c = pd.read_csv(f"{T}/smr_{key}_crude.csv").set_index("name").smr.reindex(r.name)
        y = np.arange(len(r))
        ax.axvline(1, color=INK2, lw=1)
        ax.hlines(y, r.lo, r.hi, color=[FLAGC[f_] for f_ in r.flag], lw=2.2)
        ax.scatter(r.smr_eb, y, s=36, color=[FLAGC[f_] for f_ in r.flag], zorder=3, edgecolor="white", lw=0.6,
                   label="Standardised")
        ax.scatter(c.values, y, s=26, marker="|", color=INK, zorder=4, label="Crude")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{n}  (n={int(k):,})" for n, k in zip(r.name, r.n)], fontsize=7.4)
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.5", "1", "2"])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Observed / expected deaths (log scale)")
        clean(ax, "x")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, f"{FIG}/m3_provider_type")


def m4():
    """Pandemic strain in non-COVID admissions."""
    mo = pd.read_csv(f"{T}/strain_monthly.csv")
    ld = pd.read_csv(f"{T}/strain_load.csv")
    fig = plt.figure(figsize=(10.8, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    for i, (grp, col) in enumerate((("adult non-COVID", CAT[0]), ("paediatric non-COVID", CAT[6]))):
        ax = fig.add_subplot(gs[i, 0])
        g = mo[mo.group == grp].sort_values("ym")
        g = g.assign(q=(g.ym // 3)).groupby("q").agg(O=("O", "sum"), E=("E", "sum"), ym=("ym", "min")).reset_index()
        lo = stats.chi2.ppf(0.025, 2 * g.O) / 2 / g.E
        hi = stats.chi2.ppf(0.975, 2 * (g.O + 1)) / 2 / g.E
        x = ym_to_date(g.ym.values)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=GRID, alpha=0.45, lw=0)
        ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(x, g.O / g.E, color=col, marker="o", ms=3.5)
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_ylabel("Observed / expected deaths")
        who = "Adults" if grp.startswith("adult") else "Children"
        ax.set_title(f"{who} admitted for a non-COVID ARI, by quarter "
                     f"(expected from 2015–2019 case-mix model)", fontsize=8.6)
        clean(ax)
    ax = fig.add_subplot(gs[:, 1])
    order = ["none", "0–0.5×", "0.5–1×", "1–2×", "2–4×", ">4×"]
    for strat, col, off, mk in (("adult", CAT[0], -0.12, "o"), ("non-pneumonia", CAT[2], 0.12, "s")):
        g = ld[ld.stratum == strat].set_index("load_cat").reindex(order)
        x = np.arange(len(order)) + off
        ax.errorbar(x, g.oe, yerr=[g.oe - g.lo, g.hi - g.oe], fmt=mk, color=col, ms=5, lw=1.3,
                    label={"adult": "Adults, all non-COVID ARI", "non-pneumonia": "Non-pneumonia diagnoses only"}[strat])
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7.5)
    ax.set_xlabel("COVID-19 admissions of the facility in the month,\nrelative to its pre-pandemic monthly ARI volume")
    ax.set_ylabel("Observed / expected deaths (non-COVID admissions)")
    ax.set_title("March 2020 – December 2021", fontsize=9)
    ax.legend(fontsize=7.3, loc="upper left")
    clean(ax)
    fig.tight_layout()
    save(fig, f"{FIG}/m4_pandemic_strain")


def m5():
    """Inequity decomposition."""
    t = pd.read_csv(f"{T}/inequity.csv")
    t = t[t.deaths_group >= 50]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    labels = [f"{r.contrast}\n{r.stratum.lower()}, {r.deaths_group:,} deaths" for r in t.itertuples()]
    y = np.arange(len(t))[::-1]
    for (col, lo, hi, lab, c, mk, off) in (("rr_crude", "rr_crude_lo", "rr_crude_hi", "Crude", MUTED, "o", 0.24),
                                           ("rr_clin", "rr_clin_lo", "rr_clin_hi", "Adjusted for clinical case-mix", CAT[0], "s", 0.0),
                                           ("rr_clin_prov", "rr_clin_prov_lo", "rr_clin_prov_hi",
                                            "Adjusted for clinical case-mix and place of care", CAT[1], "D", -0.24)):
        ax.errorbar(t[col], y + off, xerr=[t[col] - t[lo], t[hi] - t[col]], fmt=mk, color=c, ms=4.5, lw=1.2, label=lab)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3, 4]); ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3", "4"])
    ax.set_xlabel("Mortality ratio, group vs comparison (log scale, 95% bootstrap interval)")
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0,
              handletextpad=0.3)
    clean(ax, "x")
    ax = axes[1]
    ax.hlines(y, t.early_share_ref * 100, t.early_share_group * 100, color=GRID, lw=2)
    ax.plot(t.early_share_ref * 100, y, "o", color=MUTED, ms=6, label="Comparison group")
    ax.plot(t.early_share_group * 100, y, "D", color=CAT[6], ms=6, label="Group")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Deaths within 48 h of admission (% of in-hospital deaths)")
    ax.legend(fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    clean(ax, "x")
    fig.tight_layout()
    save(fig, f"{FIG}/m5_inequity")


def m6():
    """Post-pandemic paediatric rebound."""
    mo = pd.read_csv(f"{T}/rebound_monthly.csv")
    su = pd.read_csv(f"{T}/rebound_summary.csv")
    mo, su = mo[mo.model == "seasonal"], su[su.model == "seasonal"]
    bands = ["<1 y", "1–4 y", "5–9 y", "10–19 y"]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.6), gridspec_kw=dict(height_ratios=[1.25, 1]))
    for j, b in enumerate(bands):
        g = mo[(mo.band == b) & (mo.syndrome == "All non-COVID ARI")].sort_values("ym")
        x = ym_to_date(g.ym.values)
        ax = axes[0, j]
        ax.fill_between(x, g.lo, g.hi, color=MUTED, alpha=0.25, lw=0)
        ax.plot(x, g.expected, color=MUTED, lw=1.2, label="Expected (2015–2019 seasonal rate)")
        ax.plot(x, g.observed, color=CAT[6], lw=1.4, label="Observed")
        ax.axvline(pd.Timestamp("2020-03-01"), color=INK2, lw=0.8, ls=":")
        ax.set_title(f"Age {b}", fontsize=9.2)
        ax.set_ylim(0, None)
        ax.set_xticks(pd.to_datetime([f"{yy}-01-01" for yy in (2015, 2017, 2019, 2021, 2023)]))
        ax.set_xticklabels(["2015", "2017", "2019", "2021", "2023"])
        clean(ax)
        ax2 = axes[1, j]
        s = su[(su.band == b) & (su.syndrome == "All non-COVID ARI") & su.period.isin(["Mar 2020–Dec 2021", "2022", "2023"])]
        s = s.set_index("period").reindex(["Mar 2020–Dec 2021", "2022", "2023"])
        ax2.bar(range(3), s.oe, color=[MUTED, CAT[6], CAT[6]], width=0.6)
        ax2.errorbar(range(3), s.oe, yerr=[s.oe - s.oe_lo, s.oe_hi - s.oe], fmt="none", color=INK, lw=1)
        for k, v in enumerate(s.oe):
            ax2.text(k, s.oe_hi.iloc[k] + 0.04, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax2.axhline(1, color=INK2, lw=1, ls="--")
        ax2.set_xticks(range(3)); ax2.set_xticklabels(["Mar 2020–\nDec 2021", "2022", "2023"], fontsize=7.5)
        ax2.set_ylim(0, 2.6)
        clean(ax2)
    axes[0, 0].set_ylabel("Non-COVID ARI admissions per month")
    axes[1, 0].set_ylabel("Observed / expected")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, f"{FIG}/m6_paediatric_rebound")


def m7():
    """Syndrome-specific rebound in under-fives and the age shift of pneumonia."""
    su = pd.read_csv(f"{T}/rebound_summary.csv")
    su = su[(su.model == "seasonal") & su.period.isin(["2022", "2023"])]
    ag = pd.read_csv(f"{T}/rebound_age_shift.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    synd = ["Bronchiolitis", "Pneumonia", "Upper respiratory", "Influenza"]
    x0 = 0
    ticks, labs = [], []
    for b in ("<1 y", "1–4 y"):
        for s in synd:
            for k, (per, col) in enumerate((("2022", SEQ[2]), ("2023", SEQ[5]))):
                r = su[(su.band == b) & (su.syndrome == s) & (su.period == per)]
                if not len(r):
                    continue
                r = r.iloc[0]
                ax.bar(x0 + k * 0.38, r.oe, width=0.36, color=col, label=per if (x0 == 0) else None)
                ax.errorbar(x0 + k * 0.38, r.oe, yerr=[[r.oe - r.oe_lo], [r.oe_hi - r.oe]], fmt="none", color=INK, lw=0.9)
            ticks.append(x0 + 0.19); labs.append(f"{s}\n{b}")
            x0 += 1.1
        x0 += 0.5
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=6.6)
    ax.set_ylabel("Observed / expected admissions")
    ax.legend(fontsize=7.5)
    clean(ax)
    ax = axes[1]
    for s, col in (("Pneumonia", CAT[0]), ("Bronchiolitis", CAT[2])):
        g = ag[ag.synd == s]
        ax.plot(g.year, g.share_1plus * 100, marker="o", color=col, label=s)
    ax.axvspan(2019.5, 2021.5, color=GRID, alpha=0.45, lw=0)
    ax.set_ylabel("Admissions under 5 y that are aged ≥1 y (%)")
    ax.legend(fontsize=7.5)
    clean(ax)
    fig.tight_layout()
    save(fig, f"{FIG}/m7_syndromes_age_shift")


def m8():
    """Risk phenotypes."""
    t = pd.read_csv(f"{T}/phenotypes.csv").sort_values("death_pct")
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    y = np.arange(len(t))
    ax.barh(y, t.death_pct, color=SEQ[3], height=0.66)
    for yy, r in zip(y, t.itertuples()):
        ax.text(r.death_pct + 0.6, yy, f"{r.death_pct:.1f}%   ·   {r.share_admissions:.1%} of admissions, "
                                       f"{r.share_deaths:.1%} of deaths", va="center", fontsize=7.2, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(t.phenotype, fontsize=7)
    ax.set_xlim(0, 85)
    ax.set_xlabel("In-hospital deaths per 100 admissions (held-out 20% of the cohort)")
    clean(ax, "x")
    fig.tight_layout()
    save(fig, f"{FIG}/m8_phenotypes")


# ------------------------------------------------------------------ method figures ------------
NAME = {"reference": "Reference LR", "care_v1": "CARE v1 (equal-weight mean)", "equal_weight_prof": "Equal weight + provider profiles",
        "stack": "+ stratified stacking", "hicare": "HiCARE (+ multicalibration)", "hicare_no_year": "HiCARE without period terms",
        "hicare_lgbm_only": "HiCARE on LightGBM alone", "lgbm_prof": "LightGBM + profiles"}
MC = {"reference": COLOR["ref"], "care_v1": MUTED, "hicare": COLOR["care"], "stack": CAT[2], "equal_weight_prof": CAT[3]}


def t1():
    """Subgroup calibration: reference, CARE v1, HiCARE."""
    S = pd.read_csv(f"{T}/v2_subgroups.csv", keep_default_na=False)
    S = S[S.protocol == "cv"]
    base = S[S.model == "reference"][["variable", "level"]].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 9.2))
    y = np.arange(len(base))[::-1]
    for mod, mk, off in (("reference", "^", 0.22), ("care_v1", "o", 0.0), ("hicare", "D", -0.22)):
        g = base.merge(S[S.model == mod], on=["variable", "level"], how="left")
        if g.oe.isna().all():
            continue
        ax.plot(pd.to_numeric(g.oe), y + off, mk, color=MC[mod], ms=5, label=NAME[mod], mec="white", mew=0.5)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.axvspan(0.95, 1.05, color=GRID, alpha=0.5, lw=0)
    short = {"Unspecified acute lower respiratory infection": "Unspecified acute LRTI",
             "Suppurative and necrotic conditions": "Suppurative/necrotic LRT",
             "Acute upper respiratory infections": "Acute upper ARI"}
    ax.set_yticks(y); ax.set_yticklabels([short.get(l, l) for l in base.level], fontsize=7.4)
    prev = None
    for yy, v in zip(y, base.variable):
        if v != prev:
            ax.axhline(yy + 0.5, color=GRID, lw=0.8)
            prev = v
    ax.set_xlabel("Observed / expected deaths within the subgroup (shaded: within ±5%)")
    ax.legend(fontsize=7.5, loc="lower right")
    clean(ax, "x")
    fig.tight_layout()
    save(fig, f"{FIG}/t1_subgroup_calibration")


def t2():
    """Step-by-step contribution of the second layer (cross-validation)."""
    L = pd.read_csv(f"{T}/v2_metrics.csv")
    L = L[L.protocol == "cv"]
    mods = ["reference", "care_v1", "equal_weight_prof", "stack", "hicare"]
    mets = [("auroc", "AUROC ↑", "total"), ("logloss", "Log loss ↓", "total"), ("ici", "ICI ↓", "total"),
            ("cal_int_abs", "|Calibration intercept| ↓, children", "pediatric"),
            ("sub_mean_abs_log_oe", "Mean |log O/E| over 42 subgroups ↓", "total"),
            ("sub_max_ece", "Worst subgroup ECE ↓", "total")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (m, lab, s) in zip(axes.ravel(), mets):
        v = [L[(L.model == k) & (L.metric == m) & (L["sample"] == s)].value.mean() for k in mods]
        ax.bar(range(len(mods)), v, color=[MC.get(k, MUTED) for k in mods], width=0.62)
        fin = [x for x in v if np.isfinite(x)]
        if fin:
            pad = (max(fin) - min(fin)) * 0.25 + 1e-9
            ax.set_ylim(max(0, min(fin) - pad) if m != "auroc" else min(fin) - pad, max(fin) + pad)
        for k, x in enumerate(v):
            if np.isfinite(x):
                ax.text(k, x, f"{x:.4f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels(["Reference", "CARE v1", "+ profiles", "+ stacking", "HiCARE"], fontsize=7.2)
        ax.set_title(lab, fontsize=9)
        clean(ax)
    fig.tight_layout()
    save(fig, f"{FIG}/t2_layer_contributions")


def t3():
    """Monthly observed/expected on 2022-2023 under static, rolling and dynamic recalibration."""
    f = F.build()
    y, ym = f.dead.values.astype(float), f.adm_ym.values
    P = f"{JD}/results/v2/preds/temporal"
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    for ax, grp, title in ((axes[0], None, "All ARI admissions"), (axes[1], "covid", "COVID-19 admissions")):
        for name, lab, col, ls in (("hicare_static", "HiCARE, static", COLOR["care"], ":"),
                                   ("hicare+roll", "HiCARE + rolling window (v1)", COLOR["care"], "--"),
                                   ("hicare", "HiCARE + state-space recalibration", COLOR["care"], "-"),
                                   ("reference", "Reference, best variant", COLOR["ref"], "-")):
            fp = f"{P}/{name}.parquet"
            if not os.path.exists(fp):
                continue
            d = pd.read_parquet(fp)
            m = np.ones(len(d), bool) if grp is None else (f.ari_type.astype(str).values[d.idx.values] == "COVID-19")
            g = pd.DataFrame(dict(ym=ym[d.idx.values][m], y=y[d.idx.values][m], p=d.p.values[m])).groupby("ym").agg(
                O=("y", "sum"), E=("p", "sum"))
            g = g.groupby(g.index // 2 * 2).sum()
            ax.plot(ym_to_date(g.index.values), g.O / g.E, ls, color=col, label=lab, lw=1.7)
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_yscale("log"); ax.set_yticks([0.4, 0.6, 0.8, 1, 1.25, 1.6]); ax.set_yticklabels(["0.4", "0.6", "0.8", "1", "1.25", "1.6"])
        ax.set_title(title + ", two-month periods", fontsize=9)
        clean(ax)
    axes[0].set_ylabel("Observed / expected deaths (log scale)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, f"{FIG}/t3_dynamic_recalibration")


def t4():
    """Geographic transport with and without provider profiles."""
    L = pd.read_csv(f"{T}/v2_metrics.csv")
    L = L[(L.protocol == "geo")]
    mods = [("reference", "Reference LR", COLOR["ref"]), ("care_v1", "Location-free (v1)", MUTED),
            ("hicare", "Location-free + provider profiles", COLOR["care"]),
            ("full_with_profiles", "Identifiers + profiles", CAT[3])]
    mets = [("auroc", "AUROC ↑"), ("logloss", "Log loss ↓"), ("ici", "ICI ↓"), ("cal_slope_err", "|Calibration slope − 1| ↓")]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.4))
    for ax, (m, lab) in zip(axes, mets):
        v = [L[(L.model == k) & (L.metric == m) & (L["sample"] == "total")].value.mean() for k, _, _ in mods]
        ax.bar(range(len(mods)), v, color=[c for _, _, c in mods], width=0.62)
        fin = [x for x in v if np.isfinite(x)]
        if fin:
            pad = (max(fin) - min(fin)) * 0.3 + 1e-9
            ax.set_ylim(max(0, min(fin) - pad) if m != "auroc" else min(fin) - pad, max(fin) + pad)
        for k, x in enumerate(v):
            if np.isfinite(x):
                ax.text(k, x, f"{x:.4f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(mods))); ax.set_xticklabels(["Ref.", "Loc.-free", "+ profiles", "IDs +\nprofiles"], fontsize=7)
        ax.set_title(lab, fontsize=9)
        clean(ax)
    fig.tight_layout()
    save(fig, f"{FIG}/t4_geographic_profiles")


# =========================================================================================
# Extended figure set (x01-x20)
# =========================================================================================
PV = f"{JD}/results/v2/preds"
M3 = [("reference", "Reference LR", COLOR["ref"], "^", "-"), ("care_v1", "CARE v1 (equal weight)", MUTED, "o", "--"),
      ("hicare", "HiCARE", COLOR["care"], "D", "-")]
SNAME = {"total": "All ages", "pediatric": "Paediatric (0–19 y)", "adult": "Adult (≥20 y)"}
DIV = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", [CAT[0], "#f0efec", CAT[1]])


def panel(ax, letter, dx=-0.02, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, ha="right", va="bottom")


def _cv(name):
    f = F.build()
    d = pd.read_parquet(f"{PV}/cv/{name}.parquet").set_index("idx").sort_index()
    return f, d


def _strata(f, d):
    ped = f.pediatric.values[d.index.values]
    y = f.dead.values[d.index.values].astype(float)
    return {"total": (y, d.p.values), "pediatric": (y[ped == 1], d.p.values[ped == 1]),
            "adult": (y[ped == 0], d.p.values[ped == 0])}


def x01():
    """HiCARE architecture."""
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.axis("off")

    def box(x, y, w, h, title, body="", fc="#f6f5f2", bar=None, tf=8.6, bf=7.0):
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                                           fc=fc, ec=GRID, lw=1))
        if bar:
            ax.add_patch(plt.Rectangle((x, y), 0.008, h, fc=bar, ec="none"))
        ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=tf, fontweight="bold", color=INK)
        if body:
            ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=bf, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1, col=INK2):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1))

    box(0.00, 0.60, 0.19, 0.36, "ARI cohort", "576,195 hospitalisations\n41,048 deaths\n33 discharge-abstract\nfeatures + lagged\nepidemic context")
    box(0.00, 0.10, 0.19, 0.38, "Rest of the registry", "9,463,108 non-ARI\ndischarges\n→ provider profiles\n(19 features per facility\ngroup and month)", fc=SEQ[0])
    box(0.25, 0.66, 0.17, 0.26, "LightGBM", "boosted trees,\nnative categoricals", fc="#ffffff", bar=COLOR["lgbm"])
    box(0.25, 0.30, 0.17, 0.26, "XGBoost", "histogram trees,\ncategorical splits", fc="#ffffff", bar=COLOR["xgb"])
    ax.text(0.335, 0.20, "Level 1\none pooled model, all ages", ha="center", fontsize=7.4, color=MUTED)
    for y0 in (0.79, 0.43):
        arrow(0.19, 0.78, 0.25, y0)
        arrow(0.19, 0.29, 0.25, y0)
    box(0.47, 0.72, 0.20, 0.22, "2a  Stratified stacking", "separate weights for\nchildren and COVID-19", fc="#ffffff", bar=CAT[2])
    box(0.47, 0.44, 0.20, 0.22, "2b  Multicalibration\nboosting", "\nshallow trees on\nsubgroup variables", fc="#ffffff", bar=CAT[2])
    box(0.47, 0.16, 0.20, 0.22, "2c  Multiaccuracy\nprojection", "\nobserved = expected in\nevery subgroup level", fc="#ffffff", bar=CAT[2])
    ax.text(0.57, 0.075, "Level 2 — trained only on\nnested out-of-fold predictions", ha="center", fontsize=7.4, color=MUTED)
    arrow(0.42, 0.79, 0.47, 0.83); arrow(0.42, 0.43, 0.47, 0.81)
    arrow(0.57, 0.72, 0.57, 0.66); arrow(0.57, 0.44, 0.57, 0.38)
    box(0.72, 0.16, 0.14, 0.22, "3  State-space\nrecalibration", "\nmonthly Bayesian\nupdate (forward\nin time only)", fc=SEQ[0])
    arrow(0.67, 0.27, 0.72, 0.27)
    box(0.72, 0.60, 0.27, 0.36, "Population-health analyses", "risk-standardised mortality by\nprovince and provider\n"
        "excess deaths under pandemic strain\ndecomposition of mortality gaps\npaediatric rebound · risk phenotypes", fc="#ffffff", bar=COLOR["care"])
    box(0.90, 0.16, 0.10, 0.22, "Risk", "\ncalibrated\nwithin\nsubgroups", fc="#ffffff")
    arrow(0.86, 0.27, 0.90, 0.27); arrow(0.95, 0.38, 0.90, 0.60)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.02, 1.0)
    save(fig, f"{FIG}/x01_architecture")


def x02():
    """Nested cross-fitting."""
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.axis("off")
    cols = {"test": CAT[1], "pred": CAT[0], "train": SEQ[0]}
    ax.text(0.0, 5.75, "Outer test fold k = 5: the four inner models that produce the second layer's training data",
            fontsize=8.8, color=INK, fontweight="bold")
    for row, j in enumerate([1, 2, 3, 4]):
        yy = 4.4 - row * 1.1
        for c in range(1, 6):
            kind = "test" if c == 5 else ("pred" if c == j else "train")
            ax.add_patch(plt.Rectangle((c - 1 + 0.04, yy), 0.92, 0.8, fc=cols[kind], ec="white", lw=1.5))
            ax.text(c - 0.5, yy + 0.4, {"test": "never seen", "pred": "predicted", "train": "train"}[kind], ha="center", va="center",
                    fontsize=7.4, color="white" if kind != "train" else INK2)
        ax.text(-0.1, yy + 0.4, f"inner j = {j}", ha="right", va="center", fontsize=7.8, color=INK2)
    for c in range(1, 6):
        ax.text(c - 0.5, 5.35, f"fold {c}", ha="center", fontsize=7.8, color=INK2)
    ax.annotate("", (6.1, 2.9), (5.1, 2.9), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
    ax.text(6.2, 3.9, "Predictions for folds 1–4\n(from models that saw neither the\nrecord's fold nor fold 5)", fontsize=7.6, color=INK2, va="center")
    ax.text(6.2, 2.75, "→ train stacking, multicalibration\n    boosting and projection", fontsize=7.6, color=INK, va="center", fontweight="bold")
    ax.text(6.2, 1.65, "→ apply to fold 5, predicted by the\n    level-1 model trained on folds 1–4", fontsize=7.6, color=INK2, va="center")
    ax.set_xlim(-1.2, 9.6); ax.set_ylim(0.7, 6.1)
    save(fig, f"{FIG}/x02_nested_scheme")


def x03():
    """Provider profiles."""
    A = pd.read_csv(f"{T}/x_profile_by_class.csv", index_col=0)
    fg = pd.read_csv(f"{T}/x_profile_facility.csv")
    rho = json.load(open(f"{T}/x_profile_corr.json"))["rho"]
    cols = ["prof_mort", "prof_los", "prof_ped", "prof_old", "prof_outcanton", "prof_mort48", "prof_ch_obstetric",
            "prof_ch_injury", "prof_ch_neoplasm", "prof_ch_circulatory", "prof_ch_infect"]
    labs = ["All-cause\ndeath prop.", "Length\nof stay", "Share\naged 0–19", "Share\naged ≥65", "Out-of-canton\npatients", "Deaths\n<48 h",
            "Obstetric", "Injury", "Neoplasm", "Circulatory", "Infectious"]
    Z = (A[cols] - A[cols].mean()) / A[cols].std()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw=dict(width_ratios=[1.55, 1]))
    ax = axes[0]
    im = ax.imshow(Z.values, cmap=DIV, vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=6.6)
    ax.set_yticks(range(len(A))); ax.set_yticklabels([f"{i}  (n={int(n):,})" for i, n in zip(A.index, A.n)], fontsize=7.4)
    for i in range(len(A)):
        for j, c in enumerate(cols):
            v = A[c].iloc[i]
            ax.text(j, i, f"{v:.2f}" if c != "prof_los" else f"{v:.1f}", ha="center", va="center", fontsize=6.3, color=INK)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(A)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Profile of each facility class from its non-ARI patients (colour: z-score across classes)", fontsize=8.8)
    panel(ax, "A", dx=-0.25)
    ax = axes[1]
    pub = fg.sector.astype(str) == "Public"
    for m, col, lab in ((pub, CAT[0], "Public"), (~pub, CAT[2], "Private")):
        ax.scatter(fg.prof_mort[m] * 100 + 0.02, fg.ari_death_pct[m] + 0.02, s=6 + 14 * np.log10(fg.n[m]), color=col, alpha=0.7,
                   edgecolor="white", lw=0.4, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("All-cause deaths per 100 non-ARI discharges (profile)")
    ax.set_ylabel("Deaths per 100 ARI admissions")
    ax.set_title(f"{len(fg)} facility groups — Spearman ρ = {rho:.2f}", fontsize=8.8)
    ax.legend(fontsize=7.5, loc="upper left")
    clean(ax, "both")
    panel(ax, "B", dx=-0.1)
    fig.tight_layout()
    save(fig, f"{FIG}/x03_provider_profiles")


def x04():
    """ROC and precision-recall curves."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    L = pd.read_csv(f"{T}/v2_metrics.csv").set_index(["protocol", "sample", "model", "metric"]).value
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6))
    for name, lab, col, mk, ls in M3:
        f, d = _cv(name)
        for j, (s, (y, p)) in enumerate(_strata(f, d).items()):
            fpr, tpr, _ = roc_curve(y, p)
            st = max(1, len(fpr) // 2500)
            axes[0, j].plot(fpr[::st], tpr[::st], ls, color=col, lw=1.8, label=f"{lab}  {L.loc[('cv', s, name, 'auroc')]:.3f}")
            pr, rc, _ = precision_recall_curve(y, p)
            st = max(1, len(rc) // 2500)
            axes[1, j].plot(rc[::st], pr[::st], ls, color=col, lw=1.8, label=f"{lab}  {L.loc[('cv', s, name, 'auprc')]:.3f}")
            axes[0, j].set_title(SNAME[s], fontsize=9.4)
    for j in range(3):
        axes[0, j].plot([0, 1], [0, 1], color=GRID, lw=1)
        axes[0, j].set_xlabel("1 − specificity"); axes[1, j].set_xlabel("Recall")
        axes[0, j].legend(title="AUROC", fontsize=7, title_fontsize=7, loc="lower right")
        axes[1, j].legend(title="AUPRC", fontsize=7, title_fontsize=7, loc="upper right")
        axes[1, j].set_ylim(0, None)
        clean(axes[0, j], "both"); clean(axes[1, j], "both")
    axes[0, 0].set_ylabel("Sensitivity"); axes[1, 0].set_ylabel("Precision")
    panel(axes[0, 0], "A"); panel(axes[1, 0], "B")
    fig.tight_layout()
    save(fig, f"{FIG}/x04_roc_pr")


def _bins(y, p, k=20):
    o = np.argsort(p, kind="stable")
    e = np.linspace(0, len(p), k + 1).astype(int)
    return (np.array([p[o][e[i]:e[i + 1]].mean() for i in range(k)]), np.array([y[o][e[i]:e[i + 1]].mean() for i in range(k)]))


def x05():
    """Calibration curves."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    lim = {}
    for name, lab, col, mk, ls in M3:
        f, d = _cv(name)
        for j, (s, (y, p)) in enumerate(_strata(f, d).items()):
            x, o = _bins(y, p)
            ok = (x > 0) & (o > 0)
            axes[j].plot(x[ok], o[ok], ls, marker=mk, color=col, ms=4, lw=1.3, mec="white", mew=0.4, label=lab)
            lo, hi = lim.get(j, (1e9, 0))
            lim[j] = (min(lo, x[ok].min(), o[ok].min()), max(hi, x[ok].max(), o[ok].max()))
            axes[j].set_title(SNAME[s], fontsize=9.4)
    for j, ax in enumerate(axes):
        lo, hi = lim[j]
        ax.plot([lo * 0.7, hi * 1.4], [lo * 0.7, hi * 1.4], color=INK2, lw=1, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Predicted risk (20 quantile groups)")
        clean(ax, "both")
    axes[0].set_ylabel("Observed death proportion"); axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, f"{FIG}/x05_calibration")


def x06():
    """Decision curves."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    for name, lab, col, mk, ls in M3:
        f, d = _cv(name)
        for j, (s, (y, p)) in enumerate(_strata(f, d).items()):
            prev = y.mean()
            ts = np.geomspace(prev / 8, min(0.6, prev * 6), 50)
            axes[j].plot(ts, [MT.net_benefit(y, p, t) for t in ts], ls, color=col, lw=1.8, label=lab)
            if name == "reference":
                axes[j].plot(ts, [prev - (1 - prev) * t / (1 - t) for t in ts], color=GRID, lw=1.4, label="Flag all")
                axes[j].set_ylim(-prev * 0.04, prev * 1.02)
                axes[j].set_title(SNAME[s], fontsize=9.4)
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("Risk threshold (log scale)")
        clean(ax, "both")
    axes[0].set_ylabel("Net benefit"); axes[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    save(fig, f"{FIG}/x06_decision_curves")


def x07():
    """Win/loss map of HiCARE against the reference."""
    W = pd.read_csv(f"{T}/v2_winloss.csv")
    W = W[W.metric.isin(list(MT.HIGHER))]
    B = pd.read_csv(f"{T}/v2_boot.csv") if os.path.exists(f"{T}/v2_boot.csv") else None
    cols = [(p, s) for p in ("cv", "temporal", "geo") for s in ("total", "pediatric", "adult")]
    mets = list(MT.HIGHER)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for j, (p, s) in enumerate(cols):
        for i, m in enumerate(mets):
            r = W[(W.protocol == p) & (W["sample"] == s) & (W.metric == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            sig = ""
            if B is not None:
                b = B[(B.protocol == p) & (B["sample"] == s) & (B.metric == m)]
                if len(b) and not (b.lo.iloc[0] <= 0 <= b.hi.iloc[0]):
                    sig = "*"
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, fc=SEQ[1] if r.hicare_wins else "#f7c9b6", ec="white", lw=1.5))
            if m in ("cal_int_abs", "cal_slope_err", "ece", "ici"):
                txt = f"{r.hicare:.3f} | {r.reference:.3f}{sig}"
            else:
                txt = f"{(r.hicare - r.reference) / abs(r.reference):+.0%}{sig}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=6.6, color=INK)
            if pd.notna(r.v1_wins) and bool(r.v1_wins) != bool(r.hicare_wins):
                ax.plot(j + 0.9, i + 0.14, "o", ms=3.5, color=INK)
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(mets), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    pn = {"cv": "Cross-validation", "temporal": "Temporal 2022–23", "geo": "Unseen provinces"}
    sn = {"total": "all", "pediatric": "paed.", "adult": "adult"}
    ax.set_xticklabels([f"{pn[p]}\n{sn[s]}" for p, s in cols], fontsize=7.4)
    ax.set_yticks(np.arange(len(mets)) + 0.5); ax.set_yticklabels([MT.LABEL[m] for m in mets], fontsize=7.8)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(f"Blue: HiCARE better than the reference; orange: worse. Relative change, or the two values (HiCARE | reference) for calibration rows.\n"
                  f"* 95% bootstrap interval excludes 0.  • verdict differs from the first ensemble.  HiCARE better in "
                  f"{int(W.hicare_wins.sum())} of {len(W)}.", fontsize=7.2)
    fig.tight_layout()
    save(fig, f"{FIG}/x07_winloss")


def x08():
    """Paired bootstrap differences, cross-validation."""
    B = pd.read_csv(f"{T}/v2_boot.csv")
    B = B[B.protocol == "cv"]
    mets = list(MT.HIGHER)
    fig, axes = plt.subplots(3, 5, figsize=(12, 6.0))
    cs = {"total": CAT[0], "pediatric": CAT[6], "adult": CAT[2]}
    for ax, m in zip(axes.ravel(), mets):
        for k, s in enumerate(("total", "pediatric", "adult")):
            r = B[(B.metric == m) & (B["sample"] == s)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(r["diff"], 2 - k, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", color=cs[s], ms=5, lw=1.4)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["All", "Paed.", "Adult"], fontsize=7)
        ax.set_ylim(-0.6, 2.6)
        ax.set_title(MT.LABEL[m] + (" ↑" if MT.HIGHER[m] else " ↓"), fontsize=7.6)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.tick_params(axis="x", labelsize=6.5)
        ax.xaxis.get_offset_text().set_fontsize(6.5)
        clean(ax, "x")
    for ax in axes.ravel()[len(mets):]:
        ax.axis("off")
    axes.ravel()[len(mets)].text(0.0, 0.6, "HiCARE − reference,\npaired bootstrap 95% interval.\n↑ larger is better,\n↓ smaller is better.",
                                 fontsize=8, color=INK2, va="center")
    fig.tight_layout()
    save(fig, f"{FIG}/x08_bootstrap_forest")


def x09():
    """Trajectory of the state-space recalibration coefficients."""
    R = pd.read_csv(f"{T}/x_trajectory.csv")
    x = ym_to_date(R.ym.values)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.2))
    tlim = (pd.Timestamp("2017-12-01"), pd.Timestamp("2024-01-31"))
    specs = [("intercept", "Intercept (log-odds shift for all patients)", 0.0), ("slope", "Slope on the model's log-odds", 1.0),
             ("covid", "Additional offset for COVID-19 admissions", 0.0), ("paediatric", "Additional offset for children", 0.0)]
    for ax, (c, title, ref), letter in zip(axes.ravel()[:4], specs, "ABCD"):
        if c not in R:
            ax.axis("off"); continue
        for k, g in R.groupby("fold"):
            gx = ym_to_date(g.ym.values)
            if c == "covid" and k < 2020:
                continue
            ax.fill_between(gx, g[c] - 1.96 * g[c + "_sd"], g[c] + 1.96 * g[c + "_sd"], color=COLOR["care"], alpha=0.15, lw=0)
            ax.plot(gx, g[c], color=COLOR["care"], lw=1.7)
        ax.axhline(ref, color=INK2, lw=0.9, ls=":")
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=GRID, alpha=0.4, lw=0)
        ax.set_title(title, fontsize=8.8)
        ax.set_xlim(*tlim)
        clean(ax)
        panel(ax, letter)
    ax = axes[2, 0]
    ax.plot(x, R.oe_static, color=MUTED, lw=1.5, label="Static model")
    ax.plot(x, R.oe_dynamic, color=COLOR["care"], lw=1.7, label="With state-space recalibration")
    ax.axhline(1, color=INK2, lw=0.9, ls=":")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=GRID, alpha=0.4, lw=0)
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
    ax.set_title("Observed / expected deaths by month (log scale)", fontsize=8.8)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(*tlim)
    ax.legend(fontsize=7.5)
    clean(ax); panel(ax, "E")
    ax = axes[2, 1]
    ds = pd.read_csv(f"{T}/dynamic_selection.csv")
    labs = {"i": "intercept", "is": "+ slope", "isg": "+ COVID-19 and\npaediatric offsets"}
    w = 0.25
    for k, q in enumerate((0.001, 0.01, 0.1)):
        g = ds[np.isclose(ds.q, q)].set_index("feats").reindex(["i", "is", "isg"])
        ax.bar(np.arange(3) + (k - 1) * w, g.logloss, width=w * 0.92, color=SEQ[1 + 2 * k], label=f"drift variance {q:g}")
    ax.axhline(ds[ds.q.isna()].logloss.iloc[0], color=COLOR["ref"], lw=1.4, ls="--", label="rolling 1-month window")
    lo = ds.logloss.min()
    ax.set_ylim(lo - 0.0008, ds.logloss.max() + 0.0006)
    ax.set_xticks(range(3)); ax.set_xticklabels([labs[k] for k in ("i", "is", "isg")], fontsize=7.4)
    ax.set_title("Development years 2018–2021: log loss by setting", fontsize=8.8)
    ax.legend(fontsize=6.8, loc="upper right")
    clean(ax); panel(ax, "F")
    fig.tight_layout()
    save(fig, f"{FIG}/x09_state_trajectory")


def x10():
    """TreeSHAP with provider profiles."""
    imp = pd.read_csv(f"{T}/x_shap_importance.csv")
    grp = pd.read_csv(f"{T}/x_shap_groups.csv")
    dep = pd.read_parquet(f"{T}/x_shap_profile.parquet")
    gcol = {"demography": CAT[0], "diagnosis": CAT[1], "facility": CAT[2], "context": CAT[3], "residence": CAT[4],
            "calendar": CAT[5], "referral": CAT[6], "provider profile": CAT[7]}
    nice = {"age_cont": "Age", "fac": "Facility group (identifier)", "icd4": "ICD-10 code, 4 characters", "icd3": "ICD-10 category",
            "cant_res": "Canton of residence", "prof_mort": "Profile: all-cause death proportion", "ari_type_c": "Infection type",
            "ctx_nat_cfr": "Context: national death proportion", "sexo": "Sex", "ctx_prov_cfr": "Context: provincial death proportion",
            "same_cant": "Treated in own canton", "ctx_nat_cfr_adult": "Context: adult death proportion", "mes_ingr": "Admission month",
            "prof_old": "Profile: share aged ≥65", "prof_los": "Profile: length of stay", "prof_n": "Profile: volume",
            "prof_ped": "Profile: share aged 0–19", "prof_outcanton": "Profile: out-of-canton patients", "prof_mort48": "Profile: deaths <48 h",
            "entidad": "Managing entity", "clase": "Facility class", "cant_ubi": "Hospital canton", "etnia": "Ethnicity",
            "ctx_nat_covid": "Context: COVID-19 share", "ctx_nat_n": "Context: national volume", "ctx_prov_covid": "Context: provincial COVID-19 share",
            "ctx_fac_n": "Context: facility volume", "prov_res": "Province of residence", "ctx_prov_n": "Context: provincial volume",
            "ctx_fac_ratio": "Context: facility load", "adm_dow": "Weekday", "dia_ingr": "Day of month", "prov_ubi": "Hospital province"}
    fig = plt.figure(figsize=(11.6, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[:, 0])
    top = imp.head(20).iloc[::-1]
    ax.barh([nice.get(c, c.replace("prof_ch_", "Profile: share ").replace("_", " ")) for c in top.feature], top.mean_abs_shap,
            color=[gcol[g] for g in top.group], height=0.7)
    ax.set_xlabel("Mean |SHAP| (log-odds)"); ax.set_title("Top 20 features of the LightGBM member", fontsize=9)
    clean(ax, "x"); panel(ax, "A", dx=-0.45)
    ax = fig.add_subplot(gs[0, 1])
    g2 = grp.sort_values("share")
    ax.barh([g.capitalize() for g in g2.group], g2.share * 100, color=[gcol[g] for g in g2.group], height=0.7)
    for v, yy in zip(g2.share * 100, range(len(g2))):
        ax.text(v + 0.6, yy, f"{v:.0f}%", va="center", fontsize=7.4, color=INK2)
    ax.set_xlabel("Share of total attribution (%)"); ax.set_xlim(0, 52)
    clean(ax, "x"); panel(ax, "B", dx=-0.3)
    ax = fig.add_subplot(gs[1, 1])
    d = dep.dropna(subset=["prof_mort"])
    b = pd.qcut(d.prof_mort, 25, duplicates="drop")
    m = d.groupby(b, observed=True).agg(x=("prof_mort", "mean"), s=("shap_prof_mort", "mean"), t=("shap_profile", "mean"))
    ax.plot(m.x * 100, m.s, marker="o", ms=3.5, color=CAT[7], label="Profile: all-cause death proportion")
    ax.plot(m.x * 100, m.t, marker="s", ms=3.5, color=MUTED, label="All 19 profile features")
    ax.axhline(0, color=INK2, lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("Facility's all-cause deaths per 100 non-ARI discharges"); ax.set_ylabel("SHAP (log-odds)")
    ax.legend(fontsize=7)
    clean(ax, "both"); panel(ax, "C", dx=-0.16)
    fig.tight_layout()
    save(fig, f"{FIG}/x10_shap_profiles")


def x11():
    """Held-out provinces: discrimination and calibration by province."""
    G = pd.read_csv(f"{T}/x_geo_province.csv")
    order = G[G.model == "hicare"].sort_values("auroc").prov.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.2), sharey=True)
    y = np.arange(len(order))
    for ax, col, lab in ((axes[0], "auroc", "AUROC in the held-out province"), (axes[1], "oe", "Observed / expected deaths (log scale)")):
        piv = G.pivot(index="prov", columns="model", values=col).reindex(order)
        ax.hlines(y, piv.min(axis=1), piv.max(axis=1), color=GRID, lw=2)
        for name, lb, c, mk, _ in M3:
            lb2 = {"care_v1": "Location-free ensemble (v1)", "hicare": "HiCARE: location-free + profiles"}.get(name, lb)
            ax.plot(piv[name], y, mk, color=c, ms=5.5, mec="white", mew=0.5, label=lb2)
        ax.set_xlabel(lab)
        clean(ax, "x")
    axes[1].axvline(1, color=INK2, lw=0.9, ls=":"); axes[1].set_xscale("log")
    axes[1].set_xticks([0.5, 0.75, 1, 1.5, 2, 3]); axes[1].set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=7.6)
    axes[0].legend(fontsize=7.2, loc="lower right")
    panel(axes[0], "A", dx=-0.3); panel(axes[1], "B")
    fig.tight_layout()
    save(fig, f"{FIG}/x11_geo_by_province")


def x12():
    """Tile map of standardised mortality by province, period and age stratum."""
    r = pd.read_csv(f"{T}/smr_prov_ubi_rich.csv").sort_values("smr_eb", ascending=False)
    st = pd.read_csv(f"{T}/smr_province_strata.csv")
    pv = st.pivot_table(index="name", columns="stratum", values="smr_eb").reindex(r.name)
    pv.insert(0, "All years", r.smr_eb.values)
    cols = ["All years", "2015-2019", "2020-2021", "2022-2023", "paediatric", "adult"]
    labs = ["All years", "2015–2019", "2020–2021", "2022–2023", "Children", "Adults"]
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    Z = np.log(pv[cols].values.astype(float))
    im = ax.imshow(Z, cmap=DIV, vmin=-np.log(2.0), vmax=np.log(2.0), aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = pv[cols].values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=7.8)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(pv)), minor=True)
    ax.grid(which="minor", color="white", lw=1.6); ax.tick_params(which="minor", length=0)
    ax.axvline(0.5, color=INK2, lw=1.2); ax.axvline(3.5, color=INK2, lw=1.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=np.log([0.5, 0.75, 1, 1.5, 2]))
    cb.ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2"], fontsize=7)
    cb.set_label("Standardised mortality ratio", fontsize=7.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, f"{FIG}/x12_province_tilemap")


def x13():
    """Observed minus expected deaths by province."""
    r = pd.read_csv(f"{T}/smr_prov_ubi_rich.csv")
    r["ex"] = r.O - r.E
    r = r.sort_values("ex")
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    y = np.arange(len(r))
    ax.barh(y, r.ex, color=[FLAGC[f_] for f_ in r.flag], height=0.7)
    for yy, v in zip(y, r.ex):
        ax.text(v + (25 if v >= 0 else -25), yy, f"{v:+,.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(r.name, fontsize=7.8)
    ax.set_xlabel("Observed minus expected in-hospital deaths, 2015–2023")
    lim = np.abs(r.ex).max() * 1.2
    ax.set_xlim(-lim, lim)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FLAGC["high"], label="above expectation"), Patch(color=FLAGC["low"], label="below expectation"),
                       Patch(color=FLAGC["as expected"], label="as expected")], fontsize=7.5, loc="lower right")
    clean(ax, "x")
    fig.tight_layout()
    save(fig, f"{FIG}/x13_excess_deaths")


def x14():
    """Timeline: COVID-19 admissions, non-COVID admissions, excess mortality of non-COVID admissions."""
    d = pd.read_parquet(f"{T}/strain_records.parquet")
    g = d.groupby("ym").agg(covid=("covid", "sum"), n=("y", "size")).reset_index()
    non = d[~d.covid].groupby("ym").agg(n=("y", "size"), O=("y", "sum"), E=("e", "sum")).reset_index()
    from scipy import stats as st_
    x = ym_to_date(g.ym.values)
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.25]))
    axes[0].bar(x, g.covid / 1000, width=25, color=CAT[1])
    axes[0].set_ylabel("COVID-19 admissions\n(thousands per month)")
    axes[1].plot(ym_to_date(non.ym.values), non.n / 1000, color=CAT[0])
    axes[1].set_ylabel("Non-COVID ARI admissions\n(thousands per month)"); axes[1].set_ylim(0, None)
    lo = st_.chi2.ppf(0.025, 2 * non.O) / 2 / non.E
    hi = st_.chi2.ppf(0.975, 2 * (non.O + 1)) / 2 / non.E
    xn = ym_to_date(non.ym.values)
    axes[2].fill_between(xn, lo, hi, color=CAT[6], alpha=0.2, lw=0)
    axes[2].plot(xn, non.O / non.E, color=CAT[6])
    axes[2].axhline(1, color=INK2, lw=0.9, ls=":")
    axes[2].set_ylabel("Observed / expected deaths,\nnon-COVID admissions")
    for ax, letter in zip(axes, "ABC"):
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=GRID, alpha=0.4, lw=0)
        clean(ax); panel(ax, letter, dx=-0.09)
    fig.tight_layout()
    save(fig, f"{FIG}/x14_strain_timeline")


def x15():
    """Pandemic strain by province."""
    R = pd.read_csv(f"{T}/x_strain_province.csv")
    cj = json.load(open(f"{T}/x_strain_province_corr.json"))
    R = R[R.O >= 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw=dict(width_ratios=[1, 1.05]))
    r = R.sort_values("oe")
    y = np.arange(len(r))
    ax = axes[0]
    cols = [CAT[1] if l > 1 else (CAT[0] if h < 1 else MUTED) for l, h in zip(r.lo, r.hi)]
    ax.hlines(y, r.lo, r.hi, color=cols, lw=2)
    ax.scatter(r.oe, y, color=cols, s=30, zorder=3, edgecolor="white", lw=0.5)
    ax.axvline(1, color=INK2, lw=0.9)
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 3]); ax.set_xticklabels(["0.5", "1", "2", "3"]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y); ax.set_yticklabels([f"{p}  ({o} deaths)" for p, o in zip(r.prov, r.O)], fontsize=7.4)
    ax.set_xlabel("Observed / expected deaths, non-COVID admissions,\nMarch 2020 – December 2021 (log scale, 95% CI)")
    clean(ax, "x"); panel(ax, "A", dx=-0.42)
    ax = axes[1]
    ax.scatter(R.load, R.oe, s=10 + R.O / 6, color=CAT[6], alpha=0.75, edgecolor="white", lw=0.5)
    for q in R.itertuples():
        if q.O >= 80 or q.oe > 2.2 or q.oe < 0.75:
            ax.annotate(q.prov, (q.load, q.oe), xytext=(4, 3), textcoords="offset points", fontsize=6.6, color=INK2)
    ax.axhline(1, color=INK2, lw=0.9, ls=":")
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 3]); ax.set_yticklabels(["0.5", "1", "2", "3"]); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("COVID-19 admissions of the province, March 2020 – December 2021,\nrelative to its pre-pandemic ARI admissions over 22 months")
    ax.set_ylabel("Observed / expected deaths, non-COVID admissions")
    ax.set_title(f"Across provinces: Spearman ρ = {cj['rho']:.2f} (p = {cj['p']:.2f})", fontsize=8.8)
    clean(ax, "both"); panel(ax, "B", dx=-0.1)
    fig.tight_layout()
    save(fig, f"{FIG}/x15_strain_by_province")


def x16():
    """Tile map of admissions against the seasonal counterfactual, by age band and quarter."""
    mo = pd.read_csv(f"{T}/rebound_monthly.csv")
    mo = mo[(mo.model == "seasonal") & (mo.syndrome == "All non-COVID ARI") & (mo.year >= 2019)]
    bands = ["<1 y", "1–4 y", "5–9 y", "10–19 y", "≥20 y"]
    mo["q"] = (mo.year.astype(int)).astype(str) + " Q" + ((mo.month - 1) // 3 + 1).astype(str)
    g = mo.groupby(["band", "q"]).agg(o=("observed", "sum"), e=("expected", "sum")).reset_index()
    g["oe"] = g.o / g.e
    piv = g.pivot(index="band", columns="q", values="oe").reindex(bands)
    fig, ax = plt.subplots(figsize=(11.4, 3.3))
    im = ax.imshow(np.log(piv.values), cmap=DIV, vmin=-np.log(3), vmax=np.log(3), aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color=INK)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([c.replace(" ", "\n") for c in piv.columns], fontsize=6.8)
    ax.set_yticks(range(len(bands))); ax.set_yticklabels([f"Age {b}" for b in bands], fontsize=8)
    ax.set_xticks(np.arange(-.5, piv.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, len(bands)), minor=True)
    ax.grid(which="minor", color="white", lw=1.6); ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015, ticks=np.log([1 / 3, 0.5, 1, 2, 3]))
    cb.ax.set_yticklabels(["0.33", "0.5", "1", "2", "3"], fontsize=7); cb.outline.set_visible(False)
    cb.set_label("Observed / expected", fontsize=7.5)
    fig.tight_layout()
    save(fig, f"{FIG}/x16_rebound_tilemap")


def x17():
    """Seasonality before and after the pandemic."""
    S = pd.read_csv(f"{T}/x_seasonality.csv")
    series = ["All non-COVID ARI, 0–19 y", "Pneumonia, 1–9 y", "Bronchiolitis, <1 y"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, sname, letter in zip(axes, series, "ABC"):
        g = S[S.series == sname]
        pre = g[g.year <= 2019].pivot(index="month", columns="year", values="share") * 100
        ax.fill_between(pre.index, pre.min(axis=1), pre.max(axis=1), color=MUTED, alpha=0.25, lw=0, label="2015–2019 range")
        ax.plot(pre.index, pre.mean(axis=1), color=MUTED, lw=1.5, label="2015–2019 mean")
        for yv, col, ls in ((2022, CAT[3], "--"), (2023, CAT[6], "-")):
            h = g[g.year == yv].sort_values("month")
            ax.plot(h.month, h.share * 100, ls, color=col, lw=1.8, marker="o", ms=3, label=str(yv))
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=7.5)
        ax.set_title(sname, fontsize=9)
        ax.set_ylim(0, None)
        clean(ax); panel(ax, letter)
    axes[0].set_ylabel("Share of the year's admissions (%)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, f"{FIG}/x17_seasonality")


def x18():
    """Concentration of deaths."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    ph = pd.read_csv(f"{T}/phenotypes.csv")
    for name, lab, col, mk, ls in M3:
        f, d = _cv(name)
        for j, (s, (y, p)) in enumerate(_strata(f, d).items()):
            o = np.argsort(-p, kind="stable")
            cum = np.cumsum(y[o]) / y.sum()
            x = np.arange(1, len(y) + 1) / len(y)
            st = max(1, len(x) // 1500)
            axes[j].plot(x[::st] * 100, cum[::st] * 100, ls, color=col, lw=1.8, label=lab)
            axes[j].set_title(SNAME[s], fontsize=9.4)
    axes[0].plot(np.r_[0, ph.cum_share_admissions * 100], np.r_[0, ph.cum_share_deaths * 100], color=CAT[3], lw=1.3, marker="s", ms=3.5,
                 label="14 phenotypes (tree)")
    for ax in axes:
        ax.plot([0, 100], [0, 100], color=GRID, lw=1)
        ax.set_xlabel("Admissions, highest predicted risk first (%)")
        clean(ax, "both")
    axes[0].set_ylabel("Deaths captured (%)"); axes[0].legend(fontsize=7.2, loc="lower right")
    fig.tight_layout()
    save(fig, f"{FIG}/x18_concentration")


def x19():
    """Observed death proportion by age, infection group and period."""
    A = pd.read_csv(f"{T}/x_age_mortality.csv")
    bands = ["<1", "1–4", "5–9", "10–19", "20–39", "40–49", "50–59", "60–69", "70–79", "≥80"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    pc = {"2015–2019": MUTED, "2020–2021": CAT[1], "2022–2023": CAT[0]}
    for ax, grp, letter in zip(axes, ["Pneumonia (non-COVID)", "Other ARI", "COVID-19"], "ABC"):
        for per, col in pc.items():
            g = A[(A.grp == grp) & (A.per == per)].set_index("band").reindex(bands)
            g = g[(g.n >= 200) & (g.deaths >= 5)]
            if not len(g):
                continue
            pr = g.deaths / g.n
            se = np.sqrt(pr * (1 - pr) / g.n)
            xs = [bands.index(b) for b in g.index]
            ax.fill_between(xs, (pr - 1.96 * se).clip(lower=1e-4) * 100, (pr + 1.96 * se) * 100, color=col, alpha=0.15, lw=0)
            ax.plot(xs, pr * 100, marker="o", ms=3.5, color=col, lw=1.6, label=per)
        ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands, fontsize=6.8, rotation=35)
        ax.set_yscale("log"); ax.set_title(grp, fontsize=9.2)
        ax.set_yticks([0.1, 0.3, 1, 3, 10, 30]); ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
        ax.set_xlabel("Age (years)")
        clean(ax, "both"); panel(ax, letter)
    axes[0].set_ylabel("Deaths per 100 admissions (log scale)")
    axes[0].legend(fontsize=7.4, loc="upper left")
    fig.tight_layout()
    save(fig, f"{FIG}/x19_age_mortality")


def x20():
    """Paediatric rebound by province."""
    R = pd.read_csv(f"{T}/x_rebound_province.csv")
    r = R[R.period == "2023"].sort_values("oe")
    r22 = R[R.period == "2022"].set_index("prov").oe.reindex(r.prov)
    rp = R[R.period == "Mar 2020–Dec 2021"].set_index("prov").oe.reindex(r.prov)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    y = np.arange(len(r))
    cols = [CAT[6] if l > 1 else (CAT[0] if h < 1 else MUTED) for l, h in zip(r.lo, r.hi)]
    ax.hlines(y, r.lo, r.hi, color=cols, lw=2)
    ax.scatter(r.oe, y, color=cols, s=34, zorder=3, edgecolor="white", lw=0.5, label="2023 (95% prediction interval)")
    ax.scatter(r22.values, y, marker="|", s=40, color=INK, zorder=4, label="2022")
    ax.scatter(rp.values, y, marker="x", s=18, color=MUTED, zorder=4, label="March 2020 – December 2021")
    ax.axvline(1, color=INK2, lw=0.9)
    ax.set_xscale("log"); ax.set_xticks([0.2, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "3"]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y); ax.set_yticklabels([f"{p}  ({o:,})" for p, o in zip(r.prov, r.observed)], fontsize=7.5)
    ax.set_xlabel("Paediatric non-COVID ARI admissions, observed / expected from the province's\n2015–2019 seasonal rate (log scale)")
    ax.legend(fontsize=7.3, loc="lower right")
    clean(ax, "x")
    fig.tight_layout()
    save(fig, f"{FIG}/x20_rebound_by_province")



ALL = [m1, m2, m3, m4, m5, m6, m7, m8, t1, t2, t3, t4, x01, x02, x03, x04, x05, x06, x07, x08, x09, x10,
       x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]


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
