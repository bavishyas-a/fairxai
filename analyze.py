"""
Turn the attribution records and model predictions into the paper's tables.

Produces, in outputs/:
  table_performance.csv     accuracy/F1 per arch, mean and range across seeds
  table_fairness.csv        APG/DPG/EOG per arch, with the responsible pair
  table_attention.csv       IoU / FBAR / FAF per (arch, group, mask, correct)
  table_pairgaps.csv        per-pair attention gaps with bootstrap CIs
  table_maskeffect.csv      how the group gap changes across mask backends
  ranking_check.txt         whether architecture orderings survive seed variance

The last two are the ones that carry the scientific argument.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import metrics as M


def load_attrib() -> pd.DataFrame:
    files = sorted(glob.glob(str(C.WORK_ROOT / "attrib_*.parquet")))
    if not files:
        raise FileNotFoundError("no attribution parquet files -- run attribution.py")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def table_attention(df: pd.DataFrame) -> pd.DataFrame:
    """Mean IoU/FBAR/FAF per (arch, mask, group, correct), pooled over seeds,
    each with a bootstrap CI. Sample counts included -- their absence was one of
    the flaws in the original Table 4."""
    rows = []
    keys = ["arch", "mask", "group", "correct"]
    for k, g in df.groupby(keys):
        row = dict(zip(keys, k))
        row["n"] = len(g)
        for metric in ("iou", "fbar", "faf"):
            m, lo, hi = M.bootstrap_ci(g[metric], C.BOOTSTRAP_N, C.CI)
            row[f"{metric}_mean"] = m
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        row["mask_area_frac"] = g["mask_area_frac"].mean()
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(keys)
    out.to_csv(C.OUT_ROOT / "table_attention.csv", index=False)
    return out


def table_pairgaps(df: pd.DataFrame, reference="White") -> pd.DataFrame:
    """
    Per-pair attention gaps against a reference group, with bootstrap CIs.

    This replaces both the max-over-pairs summary and the pooled
    majority/minority contrast. If a CI contains zero, the paper reports no
    evidence of a gap for that pair -- it does not report the point estimate as
    a finding.
    """
    rows = []
    corr = df[df.correct]
    for (arch, mask), g in corr.groupby(["arch", "mask"]):
        ref = g[g.group == reference]
        if ref.empty:
            continue
        for grp, gg in g.groupby("group"):
            if grp == reference:
                continue
            for metric in ("iou", "fbar", "faf"):
                d, lo, hi = M.bootstrap_diff_ci(
                    ref[metric], gg[metric], C.BOOTSTRAP_N, C.CI)
                rows.append({
                    "arch": arch, "mask": mask, "metric": metric,
                    "pair": f"{reference}-{grp}",
                    "diff": d, "lo": lo, "hi": hi,
                    "excludes_zero": bool((lo > 0) or (hi < 0)),
                    "n_ref": len(ref), "n_grp": len(gg),
                })
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT_ROOT / "table_pairgaps.csv", index=False)
    return out


def table_mask_effect(pairgaps: pd.DataFrame) -> pd.DataFrame:
    """
    THE key table. For each arch and pair, how does the measured group gap change
    when the mask stops treating forehead/jawline (face_oval) and then hair
    (head_prox) as background?

    If gaps shrink substantially from landmark5 -> head_prox, the reported
    'minority attention falls outside the face' effect is substantially a
    measurement artifact of where the mask boundary was drawn. That is a
    publishable finding in its own right, and a more useful one than the claim
    the original manuscript tried to make.
    """
    p = pairgaps[pairgaps.metric == "fbar"]    # area-normalised; the only metric
                                               # comparable across mask backends    # scale-free, best for this contrast
    wide = p.pivot_table(index=["arch", "pair"], columns="mask", values="diff")
    for b in ("face_oval", "head_prox"):
        if b in wide and "landmark5" in wide:
            wide[f"shrink_vs_landmark5_{b}"] = 1 - (wide[b].abs() / wide["landmark5"].abs())
    wide = wide.reset_index()
    wide.to_csv(C.OUT_ROOT / "table_maskeffect.csv", index=False)
    return wide


def ranking_check(df: pd.DataFrame, metric="faf", mask="face_oval") -> str:
    """
    Does any claimed architecture ordering survive seed variance?

    For each arch we compute the White-vs-worst-group gap per seed, then check
    whether the across-seed ranges overlap. Overlapping ranges mean the paper
    cannot rank those architectures, full stop.
    """
    corr = df[(df.correct) & (df["mask"] == mask)]
    per = {}
    for (arch, seed), g in corr.groupby(["arch", "seed"]):
        ref = g[g.group == "White"][metric].mean()
        worst = g.groupby("group")[metric].mean().min()
        per.setdefault(arch, {})[seed] = float(ref - worst)

    lines = [f"Architecture gap ({metric}, mask={mask}), White vs worst group",
             "-" * 64]
    for arch, by_seed in per.items():
        s = M.seed_spread(by_seed)
        lines.append(f"{arch:34s} mean={s['mean']:.4f}  "
                     f"range=[{s['min']:.4f}, {s['max']:.4f}]  n={s['n_seeds']}")

    lines += ["", "Pairwise orderings that hold across EVERY seed:"]
    archs = list(per)
    any_ok = False
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            a, b = archs[i], archs[j]
            if M.orderings_agree(per[a], per[b]):
                hi = a if np.mean(list(per[a].values())) > np.mean(list(per[b].values())) else b
                lo = b if hi == a else a
                lines.append(f"  {hi} > {lo}   (consistent)")
                any_ok = True
    if not any_ok:
        lines.append("  NONE. Do not rank architectures in the paper.")

    txt = "\n".join(lines)
    (C.OUT_ROOT / "ranking_check.txt").write_text(txt)
    return txt


def main():
    df = load_attrib()
    print(f"loaded {len(df)} attribution rows\n")

    att = table_attention(df)
    print(att.head(12).to_string(index=False), "\n")

    pg = table_pairgaps(df)
    sig = pg[pg.excludes_zero]
    print(f"pairs with CI excluding zero: {len(sig)}/{len(pg)}")
    if len(sig):
        print(sig.head(12).to_string(index=False))
    print()

    me = table_mask_effect(pg)
    print(me.to_string(index=False), "\n")
    print(ranking_check(df))


if __name__ == "__main__":
    main()
