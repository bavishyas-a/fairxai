# Fairness + attribution pipeline (rebuild)

Every number in your paper should come out of this and be traceable to a file on
disk. Nothing gets typed into a table by hand.

## Setup

```bash
pip install torch torchvision timm grad-cam mediapipe opencv-python-headless \
            scikit-learn pandas pyarrow
```

Point `config.DATA_ROOT` at FairFace (the folder with `fairface_label_train.csv`
and the image directories). On Kaggle, add the FairFace dataset and use the
`/kaggle/input/...` path.

## Run order

```bash
python data.py            # sanity: split sizes per group, leakage check
python masks.py           # precompute face masks once  (~20 min for 4k images)
python train.py --lr-sweep    # optional but recommended, ~30 min
python train.py           # 4 archs x 3 seeds, resumable
python attribution.py     # GradCAM + metrics, resumable
python analyze.py         # writes outputs/*.csv
python test_metrics.py    # metric unit tests, run any time
```

`train.py` and `attribution.py` skip completed jobs, so when Kaggle kills the
session at 12h you just re-run the same command.

## What changed from the original manuscript, and why

**Area-normalised FBAR.** The old definition divided the summed attention inside
the mask by the summed attention outside. Those regions have different areas, so
under perfectly uniform attention the ratio equals the area ratio, not 1. With a
30% mask that is 0.43 — which would have tripped a "low FBAR" threshold on a
model showing no preference at all. `test_metrics.py` demonstrates this
explicitly. The new version divides per-pixel means, so 1.0 means "no
preference" regardless of mask size.

**Per-pair reporting, no pooling.** The old analysis pooled Black/Asian/Other
into "minority" and reported a max over pairs. The pooling test in
`test_metrics.py` builds a case where one group has a 20% error injection and
shows pooled APG at 0.100 against per-pair 0.200 — pooling halves it. That is
very likely why nothing crossed threshold last time. Analysis now runs on
FairFace's 7 raw race labels by default.

**Bootstrap CIs on every gap.** `table_pairgaps.csv` has an `excludes_zero`
column. If it is False, the paper says there is no evidence of a gap for that
pair. Point estimates without intervals are what let the old draft report
differences of 0.03 and 0.04 as if they were findings.

**Three seeds and a ranking guard.** The old fairness gaps for the three CNNs
(0.089 / 0.101 / 0.115) are almost certainly inside seed variance, which means
the architecture ranking — a headline claim — was unsupported.
`ranking_check.txt` will tell you which orderings hold across every seed. If it
says NONE, do not rank architectures.

**Three mask backends.** `landmark5` reproduces your original inner-face hull.
`face_oval` adds forehead, cheeks and jawline. `head_prox` dilates that as a
rough proxy for including hair.

This is the scientific core of the rebuild. Hair is one of the strongest visual
gender cues, and hairstyle presentation varies systematically by demographic
group. Under `landmark5`, a model attending to hair is scored as relying on
"spurious background". So the original finding — minority-group attention falls
outside the face — may be substantially an artifact of the mask boundary rather
than evidence of bias. `table_maskeffect.csv` measures exactly that: how much of
the group gap survives when hair stops counting as background.

Either outcome is publishable. If the gap survives, the finding is now robust to
the obvious confound. If it collapses, you have a methodological result showing
that a popular class of attribution-fairness metrics is confounded by mask
definition — which is more interesting, and which nobody has cleanly
demonstrated.

**Per-arch learning rates.** A single shared LR is not a controlled comparison;
it handicaps whichever architecture the value suits worst, and Swin generally
wants a lower LR than the CNNs when fine-tuning. Run `--lr-sweep` and record the
winners.

**Realistic epoch counts.** `MAX_EPOCHS` is 20 with patience 5. Fine-tuning
pretrained ImageNet models on binary gender classification converges in roughly
5–15 epochs. The 62–84 epochs in the old draft were not consistent with the task
or with Kaggle's session limits.

**Mask detection failures are reported.** `masks.precompute` returns detection
rate per demographic group. Images where FaceMesh finds no face are excluded from
every attribution statistic, and those failures are very unlikely to be uniform
across groups. That number belongs in Limitations.

**Identity leakage is checked.** FairFace draws from YFCC-100M and the same
person can appear more than once. `data.check_identity_leakage` catches exact and
near-exact reuse across splits. It will not catch "same person, different photo".
Report what it finds.

## On thresholds

There are deliberately no detection thresholds in `config.py`.

The old τ_IoU = 0.15 was calibrated on the same data it was applied to, and
landed one hundredth above the largest observed gap of 0.14. A reviewer cannot
distinguish that from a threshold reverse-engineered to sit just above the data,
and it produced a detector that detected nothing on the dataset it was designed
for.

If you want thresholds back, they have to be calibrated on the validation split
and then applied, untouched, to test. Otherwise report effect sizes with
confidence intervals and let the reader judge — which is stronger anyway, and is
what `analyze.py` does.

## Caveats to carry into the paper

`head_prox` is a dilation, not a segmentation. It will swallow some genuine
background near the head. If the hair effect turns out to be the story, replace
it with a real face-parsing model (BiSeNet trained on CelebAMask-HQ) before
submitting — a reviewer will push on this and they will be right to.

GradCAM on Swin targets `layers[-1].blocks[-1].norm1` with a reshape onto the 7×7
final grid. State this explicitly in the methods section; the CNN-vs-transformer
comparison is sensitive to the choice and it is the first thing a careful
reviewer will check.
