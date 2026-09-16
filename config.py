"""
Central configuration.

Everything that a reviewer might ask "what value did you use for X?" lives here,
so there is exactly one place to look and one place to change.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# On Kaggle the FairFace dataset is usually mounted read-only under /kaggle/input.
# On Colab, download it yourself and point DATA_ROOT at the extracted folder.
DATA_ROOT = Path("ff_datasets")      # contains fairface_label_train.csv, images
WORK_ROOT = Path("work")                   # checkpoints, masks, attribution records
OUT_ROOT = Path("outputs")                 # final tables

for _p in (WORK_ROOT, OUT_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- experiment
SEEDS = [0, 1, 2]                          # 3 seeds minimum. See README on why.

ARCHS = [
    "resnet18",
    "efficientnet_b0",
    "mobilenetv3_small_100",
    "swin_tiny_patch4_window7_224",
]

# Learning rates. A single shared LR is NOT a fair comparison: it silently
# handicaps whichever architecture the value suits worst, and transformers
# generally want a lower LR than CNNs when fine-tuning. Run `train.py --lr-sweep`
# once and replace these with the per-arch val-F1 winners before the real runs.
LR = {
    "resnet18": 3e-4,
    "efficientnet_b0": 3e-4,
    "mobilenetv3_small_100": 3e-4,
    "swin_tiny_patch4_window7_224": 1e-4,
}

BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4                        # applied via AdamW decoupled decay ONLY
MAX_EPOCHS = 12                         # fine-tuning pretrained nets converges in ~5-15
PATIENCE = 3                              # early stop on val macro-F1
IMG_SIZE = 224
NUM_WORKERS = 8

# ---------------------------------------------------------------- labels
# Explicit and fixed. Every rate metric below is defined relative to this.
POSITIVE_CLASS = "Male"                    # y = 1
GENDER_CLASSES = ["Female", "Male"]        # index == label

# FairFace's 7 race labels -> the analysis groups.
# Keep the 7 raw groups as the primary unit of analysis; RACE_TO_GROUP is only
# for the optional collapsed view. Pooling is what hid the effect last time.
RACE_LABELS = [
    "White", "Black", "Indian", "East Asian",
    "Southeast Asian", "Middle Eastern", "Latino_Hispanic",
]

RACE_TO_GROUP = {
    "White": "White",
    "Black": "Black",
    "East Asian": "Asian",
    "Southeast Asian": "Asian",
    "Indian": "Other",
    "Middle Eastern": "Other",
    "Latino_Hispanic": "Other",
}

# Which grouping the analysis runs on. "raw" = 7 FairFace races (recommended),
# "collapsed" = the 4-way mapping above.
GROUPING = "raw"

# ---------------------------------------------------------------- attribution
ATTN_PERCENTILE = 70        # keep top 30% of attention pixels for the IoU mask

# Mask backends to compute. The comparison between these IS a result:
#   landmark5  - 5-point convex hull, reproduces the original inner-face mask
#   face_oval  - MediaPipe FaceMesh outer contour: adds forehead, cheeks, jawline
#   head_prox  - face_oval dilated, a crude proxy for including hair
MASK_BACKENDS = ["landmark5", "face_oval", "head_prox"]
HEAD_PROX_DILATE_PX = 10    # ~12.5% of 224. Proxy only; see README caveat.

# Number of test images to run attribution on, per (arch, seed).
# GradCAM is cheap; SHAP is not. Set SHAP_N smaller.
ATTRIB_N = 4000
SHAP_N = 500

# ---------------------------------------------------------------- reporting
BOOTSTRAP_N = 5000
CI = 0.95

# NOTE: there are deliberately no detection thresholds here.
# See README, "On thresholds".
