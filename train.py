"""
Training. Resumable, because Kaggle sessions die at 12h and Colab disconnects.

Every (arch, seed) is an independent job writing its own checkpoint and its own
finished-marker, so re-running this script picks up where it left off. Run it as
many times as you need.

  python train.py                 # all archs, all seeds
  python train.py --arch resnet18 # one arch
  python train.py --lr-sweep      # short LR search on val, prints winners

Realistic cost: fine-tuning a pretrained net on FairFace converges in 5-15
epochs, not 60-90. Four archs x 3 seeds is a few hours on a T4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import config as C
import data as D


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism costs speed but makes the seed argument meaningful.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_model(arch: str) -> nn.Module:
    import timm
    return timm.create_model(arch, pretrained=True, num_classes=2)


def loaders(df, seed):
    tr, va, te = D.build_datasets(df)
    g = torch.Generator(); g.manual_seed(seed)
    mk = lambda ds, sh: DataLoader(
        ds, batch_size=C.BATCH_SIZE, shuffle=sh, num_workers=C.NUM_WORKERS,
        pin_memory=True, generator=g if sh else None, drop_last=False,
        persistent_workers=C.NUM_WORKERS > 0, prefetch_factor=4,
    )
    return mk(tr, True), mk(va, False), mk(te, False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return f1_score(t, p, average="macro"), (t, p)


def train_one(arch: str, seed: int, df, device, lr=None, max_epochs=None, quiet=False):
    ckpt = C.WORK_ROOT / f"{arch}_seed{seed}.pt"
    done = C.WORK_ROOT / f"{arch}_seed{seed}.done.json"
    if done.exists():
        if not quiet:
            print(f"skip {arch} seed{seed} (already done)")
        return json.loads(done.read_text())

    set_seed(seed)
    lr = lr or C.LR[arch]
    max_epochs = max_epochs or C.MAX_EPOCHS
    tr, va, _ = loaders(df, seed)

    model = build_model(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, best_ep, bad, t0 = -1.0, -1, 0, time.time()
    for ep in range(max_epochs):
        model.train()
        for x, y, _ in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        vf1, _ = evaluate(model, va, device)
        if not quiet:
            print(f"{arch} seed{seed} ep{ep:02d} val_macroF1={vf1:.4f}")
        if vf1 > best:
            best, best_ep, bad = vf1, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break

    meta = {
        "arch": arch, "seed": seed, "lr": lr,
        "best_val_macro_f1": best, "best_epoch": best_ep,
        "epochs_run": ep + 1, "wall_clock_sec": round(time.time() - t0, 1),
        "checkpoint": str(ckpt),
    }
    done.write_text(json.dumps(meta, indent=2))
    return meta


def lr_sweep(df, device, candidates=(1e-4, 3e-4, 1e-3)):
    """Short sweep on seed 0, 3 epochs each. Pick per-arch winners by val F1
    BEFORE the real runs, so a shared LR is not silently handicapping one model."""
    results = {}
    for arch in C.ARCHS:
        scores = {}
        for lr in candidates:
            for f in (C.WORK_ROOT / f"{arch}_seed0.done.json",
                      C.WORK_ROOT / f"{arch}_seed0.pt"):
                f.unlink(missing_ok=True)
            m = train_one(arch, 0, df, device, lr=lr, max_epochs=3, quiet=True)
            scores[lr] = m["best_val_macro_f1"]
            print(f"  {arch} lr={lr:g} -> {scores[lr]:.4f}")
        results[arch] = max(scores, key=scores.get)
        print(f"{arch}: best lr = {results[arch]:g}")
    (C.WORK_ROOT / "lr_sweep.json").write_text(json.dumps(results, indent=2))
    print("\nPaste into config.LR:", json.dumps(results, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr-sweep", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = D.make_splits(D.load_labels())

    if a.lr_sweep:
        lr_sweep(df, device)
        return

    archs = [a.arch] if a.arch else C.ARCHS
    seeds = [a.seed] if a.seed is not None else C.SEEDS
    for arch in archs:
        for seed in seeds:
            print(train_one(arch, seed, df, device))


if __name__ == "__main__":
    main()
