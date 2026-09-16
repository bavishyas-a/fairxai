"""
FairFace loading and splits.

Two things here matter for the paper's defensibility:
  1. The split is stratified jointly on (gender, race) so subgroup sizes are
     comparable across partitions.
  2. FairFace images are sourced from YFCC-100M and the same person can appear
     more than once. A purely random split can therefore leak identities between
     train and test. We cannot fix that without identity labels, so we detect
     near-duplicates cheaply and report how many we found. If the count is
     non-trivial, it goes in Limitations. Do not skip this.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def load_labels() -> pd.DataFrame:
    """Read FairFace's official label CSVs into one frame with a `split0` column."""
    train_csv = C.DATA_ROOT / "fairface_label_train.csv"
    val_csv = C.DATA_ROOT / "fairface_label_val.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found. Set config.DATA_ROOT to the folder that "
            "contains fairface_label_train.csv and the train/ val/ image dirs."
        )

    tr = pd.read_csv(train_csv)
    tr["split0"] = "train"
    frames = [tr]
    if val_csv.exists():
        va = pd.read_csv(val_csv)
        va["split0"] = "val"
        frames.append(va)
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns={"file": "path"})
    df["path"] = df["path"].astype(str)

    unknown = set(df["race"].unique()) - set(C.RACE_LABELS)
    if unknown:
        raise ValueError(f"Unexpected race labels in CSV: {unknown}")

    df["y"] = (df["gender"] == C.POSITIVE_CLASS).astype(int)
    df["group"] = (
        df["race"] if C.GROUPING == "raw" else df["race"].map(C.RACE_TO_GROUP)
    )
    return df


def make_splits(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Add a `split` column with values train/val/test.

    FairFace's own val set becomes our test set (a genuine held-out partition
    released by the dataset authors, which is stronger than carving one out
    ourselves). The official train set is then split 85/15 into train/val,
    stratified jointly on (gender, race).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""

    df.loc[df["split0"] == "val", "split"] = "test"

    tr_idx = df.index[df["split0"] == "train"]
    strata = df.loc[tr_idx, "gender"].astype(str) + "|" + df.loc[tr_idx, "race"].astype(str)

    for _, idx in df.loc[tr_idx].groupby(strata).groups.items():
        idx = np.asarray(list(idx))
        rng.shuffle(idx)
        n_val = max(1, int(round(0.15 * len(idx))))
        df.loc[idx[:n_val], "split"] = "val"
        df.loc[idx[n_val:], "split"] = "train"

    assert (df["split"] != "").all(), "some rows unassigned"
    return df


def attribution_subset(df: pd.DataFrame, n: int | None = None, seed: int = 0):
    """
    Indices into the reset-index test frame, balanced across demographic groups.

    Decided ONCE here and shared by masks.py and attribution.py, so that mask i
    always belongs to image i. Taking the first N rows instead would sample
    whatever order the CSV happens to be in, and would silently pair masks with
    the wrong images if the two scripts disagreed.

    Returns (sel, test_frame). `sel` is sorted and deterministic given `seed`.
    """
    n = n or C.ATTRIB_N
    test = df[df.split == "test"].reset_index(drop=True)
    groups = sorted(test["group"].unique())
    per = max(1, n // len(groups))
    rng = np.random.default_rng(seed)
    picks = []
    for g in groups:
        idx = test.index[test["group"] == g].to_numpy()
        picks.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    return np.sort(np.concatenate(picks)), test


def check_identity_leakage(df: pd.DataFrame, sample: int | None = 20000) -> dict:
    """
    Cheap near-duplicate detection via average-hash on downscaled greyscale.

    This will not catch 'same person, different photo' -- nothing cheap will --
    but it does catch exact and near-exact reuse across splits, which is the
    failure mode that most inflates test accuracy. Report the number found.
    """
    from PIL import Image

    sub = df if sample is None else df.sample(min(sample, len(df)), random_state=0)
    hashes: dict[str, list[str]] = {}
    for _, row in sub.iterrows():
        p = C.DATA_ROOT / row["path"]
        try:
            im = Image.open(p).convert("L").resize((8, 8))
        except Exception:
            continue
        a = np.asarray(im, dtype=np.float32)
        bits = (a > a.mean()).astype(np.uint8).tobytes()
        h = hashlib.md5(bits).hexdigest()
        hashes.setdefault(h, []).append(row["split"])

    cross = sum(1 for v in hashes.values() if len(set(v)) > 1)
    return {
        "n_checked": len(sub),
        "n_hash_collisions_across_splits": cross,
        "note": "report this number in Limitations",
    }


# ------------------------------------------------------------------ torch side
def build_datasets(df: pd.DataFrame):
    """Return train/val/test torch Datasets. Imported lazily so this module
    stays usable (for splits and leakage checks) without torch installed."""
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class FF(Dataset):
        def __init__(self, frame, tf):
            self.f = frame.reset_index(drop=True)
            self.tf = tf

        def __len__(self):
            return len(self.f)

        def __getitem__(self, i):
            r = self.f.iloc[i]
            img = Image.open(C.DATA_ROOT / r["path"]).convert("RGB")
            return self.tf(img), int(r["y"]), i

    return (
        FF(df[df.split == "train"], train_tf),
        FF(df[df.split == "val"], eval_tf),
        FF(df[df.split == "test"], eval_tf),
    )


if __name__ == "__main__":
    d = make_splits(load_labels())
    print(d.groupby(["split", "group", "gender"]).size().unstack(fill_value=0))
    print("\nsplit sizes:", d["split"].value_counts().to_dict())

    sel, _ = attribution_subset(d)
    print(f"\nattribution subset: {len(sel)} images, "
          f"{len(sel) // d[d.split == 'test']['group'].nunique()} per group")