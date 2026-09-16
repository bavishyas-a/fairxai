"""
Facial region masks.

Three backends, computed for every image, because the difference between them
is the most interesting result available to this study:

  landmark5  Convex hull over 5 landmarks (eye centres, nose tip, mouth
             corners). This reproduces the inner-face mask used in the original
             manuscript. It excludes forehead, cheeks, jawline AND HAIR, so any
             attention on those regions is scored as "background".

  face_oval  MediaPipe FaceMesh outer face contour. Includes forehead, cheeks
             and jawline. Still excludes hair.

  head_prox  face_oval dilated by a fixed radius. A crude proxy for "face plus
             hair". It is a proxy, not a segmentation: it will also swallow some
             true background near the head. Treat any conclusion that depends on
             it as provisional, and if the hair effect turns out to be the story,
             replace this with a real face-parsing model (BiSeNet trained on
             CelebAMask-HQ) before submitting.

Why this matters: hairstyle is one of the strongest visual cues for gender, and
hairstyle presentation varies systematically by demographic group. A metric that
counts hair as "spurious background" will report group differences that are an
artifact of where the mask boundary was drawn. Running all three backends is how
you find out whether that is what happened.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config as C
import data as D

# Canonical MediaPipe FaceMesh outer face contour, in loop order (36 points).
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Groups of FaceMesh indices whose centroid approximates each MTCNN landmark.
MTCNN_EQUIV = {
    "eye_r": [33, 133, 159, 145],
    "eye_l": [362, 263, 386, 374],
    "nose": [1],
    "mouth_r": [61],
    "mouth_l": [291],
}


class FaceMesher:
    """Thin wrapper that works with either the legacy `solutions` API or the
    newer `tasks` API, since Colab and Kaggle images disagree about which ships."""

    def __init__(self, task_model_path: str | None = None):
        self.mode = None
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=False, min_detection_confidence=0.5,
            )
            self.mode = "solutions"
            return
        except Exception:
            pass

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if task_model_path is None:
            task_model_path = "face_landmarker.task"
        if not Path(task_model_path).exists():
            raise FileNotFoundError(
                "Tasks API needs face_landmarker.task. Download once with:\n"
                "  curl -LO https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=task_model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        
        self._mesh = vision.FaceLandmarker.create_from_options(opts)
        self.mode = "tasks"

    def landmarks(self, rgb: np.ndarray) -> np.ndarray | None:
        """Return (468, 2) pixel coords, or None if no face was found."""
        h, w = rgb.shape[:2]
        if self.mode == "solutions":
            res = self._mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        import mediapipe as mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._mesh.detect(img)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def _fill(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def masks_for(lm: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Build all configured mask backends from one landmark set."""
    out: dict[str, np.ndarray] = {}

    if "landmark5" in C.MASK_BACKENDS:
        pts = np.stack([lm[idx].mean(axis=0) for idx in MTCNN_EQUIV.values()])
        out["landmark5"] = _fill(pts, shape)

    oval = None
    if {"face_oval", "head_prox"} & set(C.MASK_BACKENDS):
        oval = _fill(lm[FACE_OVAL_IDX], shape)

    if "face_oval" in C.MASK_BACKENDS:
        out["face_oval"] = oval

    if "head_prox" in C.MASK_BACKENDS:
        k = C.HEAD_PROX_DILATE_PX
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        out["head_prox"] = cv2.dilate(oval.astype(np.uint8), kern).astype(bool)

    return out


def precompute(df, out_path: Path = None, limit: int | None = None) -> dict:
    """
    Compute and cache masks for the attribution subset. Run once; attribution
    then reads from the cache, so FaceMesh is not re-run for every architecture.

    The subset comes from data.attribution_subset so that mask i and image i
    refer to the same picture. `sel` is stored in the cache and checked by
    attribution.py.
    """
    from PIL import Image

    out_path = out_path or (C.WORK_ROOT / "masks_test.npz")
    sel, test_all = D.attribution_subset(df, n=limit)
    test = test_all.loc[sel].reset_index(drop=True)

    mesher = FaceMesher()
    shape = (C.IMG_SIZE, C.IMG_SIZE)
    store = {b: [] for b in C.MASK_BACKENDS}
    found = np.zeros(len(test), dtype=bool)

    for i, r in test.iterrows():
        img = Image.open(C.DATA_ROOT / r["path"]).convert("RGB").resize(shape)
        lm = mesher.landmarks(np.asarray(img))
        if lm is None:
            for b in C.MASK_BACKENDS:
                store[b].append(np.zeros(shape, dtype=bool))
            continue
        found[i] = True
        ms = masks_for(lm, shape)
        for b in C.MASK_BACKENDS:
            store[b].append(ms[b])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(test)}  detected so far: {found[:i+1].mean():.3f}")

    packed = {b: np.packbits(np.stack(v), axis=None) for b, v in store.items()}
    np.savez_compressed(
        out_path, found=found, paths=test["path"].to_numpy(), sel=sel,
        shape=np.array(shape), n=len(test), **packed,
    )

    # Detection failure rate is a reportable number: those images are excluded
    # from every attribution statistic, and the failures are unlikely to be
    # uniformly distributed across demographic groups.
    rate_by_group = (
        test.assign(found=found).groupby("group")["found"].mean().to_dict()
    )
    area = {
        b: float(np.stack(store[b])[found].mean()) for b in C.MASK_BACKENDS
    }
    return {
        "n": len(test),
        "detect_rate_overall": float(found.mean()),
        "detect_rate_by_group": rate_by_group,
        "mean_mask_area_frac": area,
        "cache": str(out_path),
    }


def load_masks(path: Path = None) -> dict[str, np.ndarray]:
    path = path or (C.WORK_ROOT / "masks_test.npz")
    z = np.load(path, allow_pickle=True)
    n, shape = int(z["n"]), tuple(z["shape"])
    out = {"found": z["found"], "paths": z["paths"], "sel": z["sel"]}
    for b in C.MASK_BACKENDS:
        bits = np.unpackbits(z[b])[: n * shape[0] * shape[1]]
        out[b] = bits.reshape(n, *shape).astype(bool)
    return out


if __name__ == "__main__":
    d = D.make_splits(D.load_labels())
    info = precompute(d, limit=C.ATTRIB_N)
    for k, v in info.items():
        print(f"{k}: {v}")