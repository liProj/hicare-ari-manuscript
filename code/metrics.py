"""Discrimination, overall accuracy, calibration and clinical-utility metrics.

All functions take (y, p) as numpy arrays.  `HIGHER` records the direction of each metric so a
win/loss tally never has to guess it.
"""
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7

# metric -> True if larger is better; calibration targets are handled as |x - target|
HIGHER = {"auroc": True, "auprc": True, "brier": False, "ipa": True, "logloss": False,
          "cal_int_abs": False, "cal_slope_err": False, "ece": False, "ici": False,
          "sens_at_spec90": True, "nb_low": True, "nb_mid": True, "nb_high": True}
LABEL = {"auroc": "AUROC", "auprc": "AUPRC", "brier": "Brier score", "ipa": "Scaled Brier (IPA)",
         "logloss": "Log loss", "cal_int_abs": "|Calibration intercept|",
         "cal_slope_err": "|Calibration slope − 1|", "ece": "ECE (20 quantile bins)",
         "ici": "ICI", "sens_at_spec90": "Sensitivity at 90% specificity",
         "nb_low": "Net benefit (low threshold)", "nb_mid": "Net benefit (mid threshold)",
         "nb_high": "Net benefit (high threshold)"}
# decision thresholds per sample: around the prevalence, half and double
THRESH = {"total": (0.035, 0.07, 0.14), "pediatric": (0.0025, 0.005, 0.01),
          "adult": (0.07, 0.14, 0.28)}


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def logloss(y, p):
    p = clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_fit(y, p, iters=25):
    """Logistic recalibration y ~ a + b*logit(p) by Newton-Raphson; returns (a, b).
    The calibration intercept is reported with the slope fixed at 1 (calibration-in-the-large)."""
    x = logit(clip(p))
    a, b = 0.0, 1.0
    for _ in range(iters):
        mu = expit(a + b * x)
        w = mu * (1 - mu)
        g = np.array([np.sum(y - mu), np.sum((y - mu) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a, b = a + step[0], b + step[1]
        if np.max(np.abs(step)) < 1e-8:
            break
    a0 = 0.0
    for _ in range(iters):
        mu = expit(a0 + x)
        step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12)
        a0 += step
        if abs(step) < 1e-8:
            break
    return float(a0), float(b)


def ece(y, p, bins=20):
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    edges = np.linspace(0, len(p), bins + 1).astype(int)
    tot = 0.0
    for i in range(bins):
        s, e = edges[i], edges[i + 1]
        if e > s:
            tot += (e - s) * abs(ys[s:e].mean() - ps[s:e].mean())
    return float(tot / len(p))


def ici(y, p, bins=200):
    """Integrated calibration index: mean |observed - predicted| along a smooth calibration
    curve.  The curve is isotonic regression on 200 quantile bins (fast and monotone)."""
    from sklearn.isotonic import IsotonicRegression
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    edges = np.linspace(0, len(p), bins + 1).astype(int)
    xm = np.array([ps[edges[i]:edges[i + 1]].mean() for i in range(bins) if edges[i + 1] > edges[i]])
    ym = np.array([ys[edges[i]:edges[i + 1]].mean() for i in range(bins) if edges[i + 1] > edges[i]])
    w = np.array([edges[i + 1] - edges[i] for i in range(bins) if edges[i + 1] > edges[i]])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(xm, ym, sample_weight=w)
    return float(np.average(np.abs(iso.predict(xm) - xm), weights=w))


def net_benefit(y, p, t):
    n = len(y)
    pos = p >= t
    tp = np.sum(pos & (y == 1))
    fp = np.sum(pos & (y == 0))
    return float(tp / n - fp / n * t / (1 - t))


def sens_at_spec(y, p, spec=0.9):
    thr = np.quantile(p[y == 0], spec)
    return float(np.mean(p[y == 1] > thr))


def all_metrics(y, p, sample="total"):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    a0, b = calibration_fit(y, p)
    br = brier(y, p)
    prev = y.mean()
    t = THRESH[sample]
    return dict(
        auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
        brier=br, ipa=float(1 - br / (prev * (1 - prev))), logloss=logloss(y, p),
        cal_int=a0, cal_slope=b, cal_int_abs=abs(a0), cal_slope_err=abs(b - 1),
        ece=ece(y, p), ici=ici(y, p), sens_at_spec90=sens_at_spec(y, p),
        nb_low=net_benefit(y, p, t[0]), nb_mid=net_benefit(y, p, t[1]),
        nb_high=net_benefit(y, p, t[2]), oe=float(prev / p.mean()))


def fast_auc(y, p):
    """Mann-Whitney AUROC with average ranks; ~10x faster than sklearn for bootstrap loops."""
    from scipy.stats import rankdata
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def platt(z, y, iters=50):
    """Joint logistic recalibration p = expit(a + b*z) fitted by Newton-Raphson."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        mu = expit(a + b * z)
        w = mu * (1 - mu)
        g = np.array([np.sum(y - mu), np.sum((y - mu) * z)])
        H = np.array([[np.sum(w), np.sum(w * z)], [np.sum(w * z), np.sum(w * z * z)]])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a, b = a + step[0], b + step[1]
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(a), float(b)
