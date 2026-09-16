"""
Fetch FairFace from Hugging Face and write it out in the layout data.py expects.

Only needed if the official Google Drive download is unavailable. Produces:

    <DATA_ROOT>/fairface_label_train.csv
    <DATA_ROOT>/fairface_label_val.csv
    <DATA_ROOT>/train/0.jpg ...
    <DATA_ROOT>/val/0.jpg ...

with the same column names and string label values as the official CSVs, so
nothing downstream needs to change.

    uv add datasets pillow pandas tqdm
    uv run python fetch_data.py

~550MB for the 0.25 config. Takes a while, mostly writing images.
"""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import config as C

# The HF release stores age/gender/race as ClassLabel integers. We write the
# STRING values so the CSVs match the official ones and config.POSITIVE_CLASS
# ("Male") keeps working. Note the HF gender ordering is 0=Male, 1=Female --
# if you ever index those integers directly, do not assume 1 means male.
PADDING = "0.25"      # matches the original experiments; 1.25 is the wider crop


def dump(split_hf: str, split_dir: str, csv_name: str):
    ds = load_dataset("HuggingFaceM4/FairFace", PADDING, split=split_hf)
    feats = ds.features
    out_dir = C.DATA_ROOT / split_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ex in enumerate(tqdm(ds, desc=split_dir)):
        rel = f"{split_dir}/{i}.jpg"
        path = C.DATA_ROOT / rel
        if not path.exists():
            ex["image"].convert("RGB").save(path, quality=95)
        rows.append({
            "file": rel,
            "age": feats["age"].int2str(ex["age"]),
            "gender": feats["gender"].int2str(ex["gender"]),
            "race": feats["race"].int2str(ex["race"]),
            "service_test": ex.get("service_test", False),
        })

    df = pd.DataFrame(rows)
    df.to_csv(C.DATA_ROOT / csv_name, index=False)
    print(f"{csv_name}: {len(df)} rows")
    print(df["race"].value_counts().to_string(), "\n")


if __name__ == "__main__":
    C.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dump("train", "train", "fairface_label_train.csv")
    dump("validation", "val", "fairface_label_val.csv")
    print("done -- now run: uv run python data.py")
