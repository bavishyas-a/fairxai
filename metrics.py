"""
Fairness and attribution metrics. Pure numpy so it can be tested without a GPU
or a trained model -- see test_metrics.py.

Design decisions worth defending in the paper:

* Every gap is reported PER PAIR of groups, not as a max over pairs and not with
  minority groups pooled. A max hides which pair drives it; pooling averages
  disparities away. The max is still reported, but as a summary of the matrix,
  never as a substitute for it.

* FBAR is area-normalised: mean attention per pixel inside the mask over mean
  attention per pixel outside. Under spatially uniform attention this equals 1
  regardless of how big the mask is. The un-normalised version does not, which
  makes its value uninterpretable and its threshold mask-dependent.

* `face_attention_fraction` is reported alongside IoU because IoU depends on the
  relative areas of the mask and the thresholded attention blob, so it moves when
  you change the mask backend even if the model's behaviour is identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ fairness
def group_rates(y_true, y_pred, groups) -> pd.DataFrame:
    """Per-group n, accuracy, positive rate, TPR, FPR."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    rows = []
    for g in pd.unique(groups):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        pos = yt == 1
        neg = ~pos
        rows.append({
            "group": g,
            "n": int(m.sum()),
            "n_pos": int(pos.sum()),
            "acc": float((yt == yp).mean()) if m.sum() else np.nan,
            "pos_rate": float((yp == 1).mean()) if m.sum() else np.nan,
            "tpr": float((yp[pos] == 1).mean()) if pos.sum() else np.nan,
            "fpr": float((yp[neg] == 1).mean()) if neg.sum() else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def pairwise_gaps(rates: pd.DataFrame, col: str) -> pd.DataFrame:
    """Signed matrix of row_group - col_group for the given rate column."""
    v = rates[col]
    g = list(v.index)
    m = np.array([[v[a] - v[b] for b in g] for a in g])
    return pd.DataFrame(m, index=g, columns=g)


def max_gap(rates: pd.DataFrame, col: str) -> tuple[float, tuple[str, str]]:
    """Largest absolute pairwise gap, with the pair that produced it."""
    m = pairwise_gaps(rates, col).abs()
    flat = m.stack()
    pair = flat.idxmax()
    return float(flat.max()), pair


def fairness_summary(y_true, y_pred, groups) -> dict:
    """APG / DPG / EOG, each with the responsible pair, plus the full matrices."""
    r = group_rates(y_true, y_pred, groups)
    apg, apg_pair = max_gap(r, "acc")
    dpg, dpg_pair = max_gap(r, "pos_rate")
    eog, eog_pair = max_gap(r, "tpr")
    return {
        "rates": r,
        "APG": apg, "APG_pair": apg_pair,
        "DPG": dpg, "DPG_pair": dpg_pair,
        "EOG": eog, "EOG_pair": eog_pair,
        "matrices": {
            "acc": pairwise_gaps(r, "acc"),
            "pos_rate": pairwise_gaps(r, "pos_rate"),
            "tpr": pairwise_gaps(r, "tpr"),
        },
    }


# ------------------------------------------------------------------ attribution
def _prep(attn: np.ndarray) -> np.ndarray:
    """Non-negative attention with any NaNs zeroed."""
    a = np.nan_to_num(np.asarray(attn, dtype=np.float64), nan=0.0)
    return np.clip(a, 0, None)


def iou_at_percentile(attn: np.ndarray, mask: np.ndarray, pct: float = 70.0) -> float:
    """IoU between the top-(100-pct)% attention pixels and the mask."""
    a = _prep(attn)
    if not mask.any():
        return np.nan
    thr = np.percentile(a, pct)
    hot = a >= thr
    union = np.logical_or(hot, mask).sum()
    if union == 0:
        return np.nan
    return float(np.logical_and(hot, mask).sum() / union)


def fbar(attn: np.ndarray, mask: np.ndarray) -> float:
    """
    Area-normalised face-to-background attention ratio.

    (mean attention per pixel inside mask) / (mean attention per pixel outside).
    Equals 1.0 under uniform attention for ANY mask area, which is the whole
    point of normalising. Returns nan when either region is empty.
    """
    a = _prep(attn)
    inside, outside = mask, ~mask
    n_in, n_out = inside.sum(), outside.sum()
    if n_in == 0 or n_out == 0:
        return np.nan
    mu_out = a[outside].sum() / n_out
    if mu_out <= 0:
        return np.nan
    return float((a[inside].sum() / n_in) / mu_out)


def face_attention_fraction(attn: np.ndarray, mask: np.ndarray) -> float:
    """Share of total attention mass falling inside the mask. Scale-free."""
    a = _prep(attn)
    tot = a.sum()
    if tot <= 0:
        return np.nan
    return float(a[mask].sum() / tot)


def attribution_record(attn, mask, pct=70.0) -> dict:
    return {
        "iou": iou_at_percentile(attn, mask, pct),
        "fbar": fbar(attn, mask),
        "faf": face_attention_fraction(attn, mask),
        "mask_area_frac": float(mask.mean()),
    }


# ------------------------------------------------------------------ uncertainty
def bootstrap_ci(values, n_boot=5000, ci=0.95, seed=0) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap CI."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_diff_ci(a, b, n_boot=5000, ci=0.95, seed=0):
    """
    CI for mean(a) - mean(b) by independent resampling.

    If this interval contains 0, you do not have evidence of a group difference,
    and the paper should say so rather than reporting the point estimate as a
    finding. This is the check that was missing last time.
    """
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    d = da - db
    lo, hi = np.percentile(d, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def seed_spread(values_by_seed: dict[int, float]) -> dict:
    """
    Mean and range across seeds for a single scalar.

    Use this before claiming any architecture ordering. If the across-seed range
    for two architectures overlaps, they are not distinguishable by this
    experiment and the paper must not rank them.
    """
    v = np.array(list(values_by_seed.values()), dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        "min": float(v.min()),
        "max": float(v.max()),
        "n_seeds": len(v),
    }


def orderings_agree(a_by_seed: dict, b_by_seed: dict) -> bool:
    """True only if every seed puts a above b. Cheap guard against the
    single-seed ranking mistake."""
    ka = set(a_by_seed) & set(b_by_seed)
    return all(a_by_seed[k] > b_by_seed[k] for k in ka) or \
           all(a_by_seed[k] < b_by_seed[k] for k in ka)
