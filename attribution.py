"""
GradCAM attribution over the test subset, for every (arch, seed, mask backend).

The Swin adaptation is the fiddly part and the one a reviewer will check, so it
is explicit here rather than buried:

  * CNNs: target the last conv block. Activations arrive as (B, C, H, W) and need
    no reshaping.
  * Swin: there is no conv layer at that depth. We target the final block's
    norm1, whose activations are token-shaped. timm has used both (B, L, C) and
    (B, H, W, C) layouts across versions, so `swin_reshape` handles either and
    asserts the grid is square. At 224px with patch 4 and 4 stages the final grid
    is 7x7.

Writes one row per (image, arch, seed, mask_backend) to a parquet file so the
analysis is a pure table operation afterwards.

  pip install grad-cam
  python attribution.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as C
import data as D
import masks as MK
import metrics as M


def swin_reshape(tensor, height=None, width=None):
    """Map Swin token activations onto their spatial grid."""
    t = tensor
    if t.dim() == 4:            # (B, H, W, C) -- newer timm
        return t.permute(0, 3, 1, 2).contiguous()
    if t.dim() == 3:            # (B, L, C)
        b, l, c = t.shape
        h = height or int(round(l ** 0.5))
        w = width or (l // h)
        assert h * w == l, f"token count {l} is not a {h}x{w} grid"
        return t.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"unexpected activation shape {tuple(t.shape)}")


def target_layer_and_reshape(model, arch):
    """Return (layers, reshape_fn) for pytorch-grad-cam."""
    if arch.startswith("swin"):
        block = model.layers[-1].blocks[-1]
        return [block.norm1], swin_reshape
    if arch.startswith("resnet"):
        return [model.layer4[-1]], None
    if arch.startswith("efficientnet") or arch.startswith("mobilenet"):
        # timm exposes the final conv stage as conv_head / blocks[-1]
        layer = getattr(model, "conv_head", None)
        return [layer if layer is not None else model.blocks[-1]], None
    raise ValueError(f"no target layer rule for {arch}")


def run(arch: str, seed: int, df, mask_store, device, limit=None):
    import timm
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from torch.utils.data import DataLoader

    ckpt = C.WORK_ROOT / f"{arch}_seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing -- run train.py first")

    model = timm.create_model(arch, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()

    layers, reshape = target_layer_and_reshape(model, arch)
    cam = GradCAM(model=model, target_layers=layers, reshape_transform=reshape)

    # Same subset the masks were built for. The guard turns a silent
    # mask/image misalignment -- which would produce plausible but entirely
    # wrong numbers -- into a loud failure.
    sel, test_all = D.attribution_subset(df, n=limit)
    if "sel" in mask_store and not np.array_equal(mask_store["sel"], sel):
        raise RuntimeError(
            "mask cache was built for a different subset. Delete "
            f"{C.WORK_ROOT / 'masks_test.npz'} and re-run masks.py."
        )

    _, _, test_ds = D.build_datasets(df)
    loader = DataLoader(torch.utils.data.Subset(test_ds, sel.tolist()),
                        batch_size=16, shuffle=False, num_workers=C.NUM_WORKERS)

    test_rows = test_all.loc[sel].reset_index(drop=True)
    found = mask_store["found"]

    rows = []
    off = 0
    for x, y, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).cpu().numpy()
        for j in range(len(x)):
            i = off + j
            if not found[i]:
                continue
            grays = cam(input_tensor=x[j:j + 1],
                        targets=[ClassifierOutputTarget(int(pred[j]))])
            attn = grays[0]
            r = test_rows.iloc[i]
            base = {
                "arch": arch, "seed": seed, "idx": i,
                "group": r["group"], "gender": r["gender"],
                "y": int(r["y"]), "pred": int(pred[j]),
                "correct": bool(int(r["y"]) == int(pred[j])),
            }
            for b in C.MASK_BACKENDS:
                rec = M.attribution_record(attn, mask_store[b][i], C.ATTN_PERCENTILE)
                rows.append({**base, "mask": b, **rec})
        off += len(x)
        if off % 640 == 0:
            print(f"  {arch} seed{seed}: {off}/{len(sel)}")

    out = C.WORK_ROOT / f"attrib_{arch}_seed{seed}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")
    df = D.make_splits(D.load_labels())
    store = MK.load_masks()

    for arch in ([a.arch] if a.arch else C.ARCHS):
        for seed in ([a.seed] if a.seed is not None else C.SEEDS):
            out = C.WORK_ROOT / f"attrib_{arch}_seed{seed}.parquet"
            if out.exists():
                print(f"skip {out.name}")
                continue
            run(arch, seed, df, store, device, a.limit)


if __name__ == "__main__":
    main()