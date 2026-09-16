"""
Tests for metrics.py. These check the properties the paper's claims depend on,
against synthetic inputs whose correct answers are known by construction.

Run: python test_metrics.py
"""
import numpy as np

import metrics as M


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_fbar_uniform_is_one_for_any_mask_area():
    """The reason for area-normalising: uniform attention gives 1.0 whatever
    fraction of the image the mask covers."""
    attn = np.ones((100, 100))
    for frac in (0.05, 0.2, 0.5, 0.9):
        m = np.zeros((100, 100), bool)
        k = int(round(frac * 100))
        m[:k, :] = True
        approx(M.fbar(attn, m), 1.0)
    print("ok  fbar uniform == 1.0 across mask areas 5%-90%")


def test_fbar_detects_concentration():
    """All mass inside a 10% mask -> per-pixel ratio is large; all outside -> 0."""
    m = np.zeros((100, 100), bool)
    m[:10, :] = True
    inside_only = np.zeros((100, 100)); inside_only[:10, :] = 1.0
    assert np.isnan(M.fbar(inside_only, m)), "zero background should be nan"

    mixed = np.full((100, 100), 1.0)
    mixed[:10, :] = 9.0
    approx(M.fbar(mixed, m), 9.0)
    print("ok  fbar recovers a known 9:1 per-pixel concentration")


def test_unnormalised_fbar_would_have_been_misleading():
    """Demonstrates the bug in the original definition: the raw sum ratio under
    uniform attention just reports the mask area ratio."""
    attn = np.ones((100, 100))
    m = np.zeros((100, 100), bool)
    m[:30, :] = True
    raw = attn[m].sum() / attn[~m].sum()
    approx(raw, 30 / 70)          # 0.43 -- would trip a 'low FBAR' threshold
    approx(M.fbar(attn, m), 1.0)  # normalised: correctly says 'no preference'
    print("ok  raw sum-ratio would flag uniform attention; normalised does not")


def test_iou_perfect_and_disjoint():
    a = np.zeros((100, 100))
    a[40:60, 40:60] = 1.0          # top 4% of pixels are the hot square
    m = np.zeros((100, 100), bool)
    m[40:60, 40:60] = True
    # At pct=70 the threshold sits at 0, so 'hot' is the whole image.
    # Use a high percentile so the hot set is the square itself.
    approx(M.iou_at_percentile(a, m, pct=96.0), 1.0)

    m2 = np.zeros((100, 100), bool)
    m2[:10, :10] = True
    approx(M.iou_at_percentile(a, m2, pct=96.0), 0.0)
    print("ok  iou == 1.0 on perfect overlap, 0.0 on disjoint")


def test_face_attention_fraction():
    a = np.zeros((10, 10)); a[0, :] = 3.0; a[1, :] = 1.0
    m = np.zeros((10, 10), bool); m[0, :] = True
    approx(M.face_attention_fraction(a, m), 30 / 40)
    print("ok  face_attention_fraction matches hand computation")


def test_fairness_metrics_known_case():
    """Two groups, hand-built confusion structure."""
    # Group A: 10 pos all predicted pos, 10 neg all predicted neg -> acc 1.0
    # Group B: 10 pos, 5 predicted pos; 10 neg all predicted neg -> acc 0.75
    y_true = np.array([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
    y_pred = np.array([1] * 10 + [0] * 10 + [1] * 5 + [0] * 5 + [0] * 10)
    grp = np.array(["A"] * 20 + ["B"] * 20)

    s = M.fairness_summary(y_true, y_pred, grp)
    r = s["rates"]
    approx(r.loc["A", "acc"], 1.0)
    approx(r.loc["B", "acc"], 0.75)
    approx(r.loc["A", "tpr"], 1.0)
    approx(r.loc["B", "tpr"], 0.5)
    approx(s["APG"], 0.25)
    approx(s["EOG"], 0.5)
    approx(s["DPG"], 0.25)   # pos_rate 0.5 vs 0.25
    print("ok  APG/DPG/EOG match hand computation, with pairs identified")


def test_pooling_hides_disparity():
    """The failure mode from the original paper: pooling minorities averages a
    real gap away."""
    rng = np.random.default_rng(0)
    n = 2000
    # White fine; Black badly served; Asian slightly better than White.
    y_true = rng.integers(0, 2, n * 3)
    grp = np.array(["White"] * n + ["Black"] * n + ["Asian"] * n)
    y_pred = y_true.copy()
    # inject 20% errors for Black only
    bidx = np.where(grp == "Black")[0]
    flip = rng.choice(bidx, size=int(0.20 * n), replace=False)
    y_pred[flip] = 1 - y_pred[flip]

    per_pair = M.fairness_summary(y_true, y_pred, grp)
    pooled_grp = np.where(grp == "White", "majority", "minority")
    pooled = M.fairness_summary(y_true, y_pred, pooled_grp)

    assert per_pair["APG"] > pooled["APG"] * 1.5, (per_pair["APG"], pooled["APG"])
    print(f"ok  per-pair APG {per_pair['APG']:.3f} vs pooled {pooled['APG']:.3f} "
          f"-- pooling roughly halves it")


def test_bootstrap_diff_covers_zero_when_no_effect():
    rng = np.random.default_rng(1)
    a = rng.normal(0.70, 0.10, 400)
    b = rng.normal(0.70, 0.10, 400)
    d, lo, hi = M.bootstrap_diff_ci(a, b, n_boot=2000)
    assert lo < 0 < hi, (d, lo, hi)

    c = rng.normal(0.55, 0.10, 400)
    d2, lo2, hi2 = M.bootstrap_diff_ci(a, c, n_boot=2000)
    assert lo2 > 0, (d2, lo2, hi2)
    print("ok  bootstrap CI covers 0 under no effect, excludes 0 under a real one")


def test_ordering_guard():
    a = {0: 0.10, 1: 0.09, 2: 0.11}
    b = {0: 0.08, 1: 0.12, 2: 0.07}   # overlapping -> not a real ordering
    assert not M.orderings_agree(a, b)
    c = {0: 0.02, 1: 0.03, 2: 0.01}
    assert M.orderings_agree(a, c)
    print("ok  ordering guard rejects overlapping architectures")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nall metric tests passed")
