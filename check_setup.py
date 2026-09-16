"""
Setup check. Run this before anything else:

    uv run python check_setup.py

Finds the FairFace label CSVs anywhere under the project, verifies the image
folders sit alongside them, checks the columns, and prints the exact DATA_ROOT
line to paste into config.py. Also warns about local directories that shadow
installed packages -- a folder named `datasets/` will break `import datasets`,
and one named `data/` will break `import data` from this project.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SHADOW_NAMES = {
    "datasets": "the HuggingFace `datasets` library",
    "data": "this project's own data.py",
    "config": "this project's own config.py",
    "metrics": "this project's own metrics.py",
    "masks": "this project's own masks.py",
}

REQUIRED_COLS = {"file", "age", "gender", "race"}


def check_shadowing(root: Path):
    problems = []
    for p in root.iterdir():
        if p.is_dir() and p.name in SHADOW_NAMES and not (p / "__init__.py").exists():
            problems.append((p, SHADOW_NAMES[p.name]))
    return problems


def find_csvs(root: Path, max_depth=4):
    hits = []
    for p in root.rglob("fairface_label_train.csv"):
        if len(p.relative_to(root).parts) <= max_depth:
            hits.append(p)
    return hits


def describe(csv: Path):
    import pandas as pd
    folder = csv.parent
    df = pd.read_csv(csv, nrows=5)
    cols = set(df.columns)
    info = {
        "folder": folder,
        "n_cols_ok": REQUIRED_COLS.issubset(cols),
        "missing_cols": sorted(REQUIRED_COLS - cols),
        "has_val_csv": (folder / "fairface_label_val.csv").exists(),
        "sample_file_value": str(df["file"].iloc[0]) if "file" in cols else None,
    }
    # The `file` column is relative, e.g. "train/1.jpg". Check it resolves.
    if info["sample_file_value"]:
        info["image_resolves"] = (folder / info["sample_file_value"]).exists()
        info["subdirs_present"] = sorted(
            d.name for d in folder.iterdir() if d.is_dir()
        )[:10]
    return info


def main():
    root = Path.cwd()
    print(f"project root: {root}\n")

    shadows = check_shadowing(root)
    if shadows:
        print("!! DIRECTORY SHADOWING -- these will break imports:")
        for p, what in shadows:
            print(f"   {p.name}/  shadows {what}")
            print(f"   fix:  mv {p.name} ff_{p.name}")
        print()

    hits = find_csvs(root)
    if not hits:
        print("no fairface_label_train.csv found under this folder.")
        print("Download it from https://github.com/dchen236/FairFace")
        print("(Labels: Train / Validation, plus the padding=0.25 image zip)")
        return 1

    print(f"found {len(hits)} candidate location(s):\n")
    good = []
    for csv in hits:
        info = describe(csv)
        print(f"  {info['folder']}")
        print(f"    columns ok       : {info['n_cols_ok']}"
              + (f"  missing {info['missing_cols']}" if info["missing_cols"] else ""))
        print(f"    val csv present  : {info['has_val_csv']}")
        print(f"    sample file value: {info['sample_file_value']}")
        print(f"    image resolves   : {info.get('image_resolves')}")
        print(f"    subdirs          : {info.get('subdirs_present')}")
        print()
        if info["n_cols_ok"] and info.get("image_resolves"):
            good.append(info["folder"])

    if good:
        rel = good[0].relative_to(root) if good[0].is_relative_to(root) else good[0]
        print("Paste this into config.py:\n")
        print(f'    DATA_ROOT = Path("{rel}")\n')
        print("then:  uv run python data.py")
        return 0

    print("CSVs found but images did not resolve next to them.")
    print("The `file` column is relative to the CSV's folder, so train/ and val/")
    print("must sit in the SAME directory as fairface_label_train.csv.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
