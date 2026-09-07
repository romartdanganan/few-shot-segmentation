# Check-in Notes — 10/09 (Group B, Junhong)

## Progress

- `main` merged with the Week 6 training work (was sitting unmerged on a
  feature branch — see "Issues" below).
- First real end-to-end training runs completed, k=5, 10 epochs each,
  seed 0:

| Method               | Best val mIoU | F1 (same epoch) |
|----------------------|---------------|------------------|
| Baseline (fine-tune) | 0.8409        | 0.8470           |
| Prototype (episodic) | 0.7926        | 0.8059           |

- Both use the identical SegFormer MiT-B0 backbone (ADE20K-pretrained),
  identical data/augmentation/optimiser, per the Table I matched
  configuration in the design report — so the gap is attributable to the
  training objective, not a confound.
- Class-level train/val/test splits confirmed with no leakage between
  seen and novel classes.

## How to read this result

- Design report predicted prototype > baseline at k=1, gap narrowing as
  k increases to 5. What we're seeing at k=5 (baseline ahead) is
  consistent with that trend, not a contradiction — fine-tuning gets
  real gradient steps on 5 labelled examples, so it's expected to close
  the gap or overtake as k grows.
- k=1 runs are next (see below) — that's the actual test of the
  hypothesis.

## Technical issues encountered

- The last month's training-pipeline work (k-shot adaptation, backbone
  matched to the actual ADE20K checkpoint, held-out query validation)
  was completed but sat on an unmerged branch instead of `main` —
  caught and merged this week. Repo history now accurately reflects
  progress.

## Next steps

- k=1 runs for both methods (direct test of the design report's
  hypothesis)
- Distance-weighted prototype ablation (`--weighted` flag) — testing
  whether down-weighting low-confidence boundary pixels during
  prototype computation improves on the plain PANet-style average
- Multi-seed evaluation via `src/evaluate.py` (mean ± std across
  episodes, not just single-seed validation numbers)
- Qualitative examples and per-class error analysis
- Literature framing: comparing against PANet/PFENet-style prototype
  averaging, with the weighted ablation as the concrete point of
  difference

## Anticipated questions

**What is your baseline?**
Direct supervised fine-tuning of the shared SegFormer MiT-B0 encoder
plus a 1x1 conv segmentation head, trained with cross-entropy directly
on the k-shot support set. At evaluation time it's further adapted
per-episode with a few extra gradient steps on that episode's support
set (`adapt_baseline`, 5 steps) before being scored on the held-out
query image.

**How many baselines/methods do you have?**
One baseline (fine-tuning) and one proposed method (prototype-based
episodic training), with a controlled ablation on the proposed method
(plain vs. distance-weighted prototypes) — three conditions total,
all sharing the same backbone and data.

**Why SegFormer MiT-B0?**
Best accuracy-to-parameter trade-off among SegFormer variants; larger
variants (B3+) don't comfortably fit an 8GB RTX 3070's memory budget
once optimiser state and activations are included.

**Why FSS-1000?**
Purpose-built for this setting — 1000 classes, official base/novel
split, large enough for real per-class error analysis, unlike
few-shot benchmarks that repurpose a handful of PASCAL VOC classes.

**How do you prevent seen/novel leakage?**
Class-level splits — no class appears in more than one of
train/val/test.

**What metrics are you reporting?**
mIoU and F1, mean ± std across multiple sampled episodes/seeds (once
the full `evaluate.py` protocol runs, not just the single-seed
validation loop inside training).
