# Check-in Notes — 10/09 (Group B, Junhong)

## Progress

- `main` merged with the Week 6 training work (was sitting unmerged on a
  feature branch — see "Issues" below).
- Five real end-to-end training runs completed (10 epochs each, seed 0),
  results archived in `results/*.json`:

| Method                | k | Best val mIoU | F1 (same epoch) |
|------------------------|---|---------------|------------------|
| Baseline (fine-tune)   | 5 | 0.8409        | 0.8470           |
| Prototype (plain)      | 5 | 0.7926        | 0.8059           |
| Prototype (weighted)   | 5 | 0.7914        | 0.8057           |
| Baseline (fine-tune)   | 1 | 0.8262        | 0.8356           |
| Prototype (plain)      | 1 | 0.7556        | 0.7130           |

- All runs use the identical SegFormer MiT-B0 backbone (ADE20K-pretrained),
  identical data/augmentation/optimiser, per the Table I matched
  configuration in the design report — so gaps are attributable to the
  training objective, not a confound.
- Class-level train/val/test splits confirmed with no leakage between
  seen and novel classes.

## How to read these results (say this plainly, don't oversell)

- **Baseline currently beats prototype at both k=1 and k=5** — this goes
  against the design report's hypothesis that prototype should win at
  low k. Not hiding this; it's a real, single-seed finding.
- **These are single-seed (seed=0) results.** Few-shot performance is
  known to be sensitive to which support examples get sampled — this
  is exactly why the design report's evaluation protocol calls for
  averaging mIoU/F1 across multiple seeds via `src/evaluate.py`, not
  single-run validation numbers. Conclusions are provisional until that
  multi-seed pass is done — flagged as the top next step.
- **Weighted vs. plain prototype at k=5 is a 0.001 mIoU difference**
  (0.7914 vs 0.7926) — within noise on one seed, no real signal yet
  either way.

## Technical issues encountered

- The last month's training-pipeline work (k-shot adaptation, backbone
  matched to the actual ADE20K checkpoint, held-out query validation)
  was completed but sat on an unmerged branch instead of `main` —
  caught and merged this week. Repo history now accurately reflects
  progress.

## Next steps

- Multi-seed evaluation via `src/evaluate.py` (mean ± std across
  episodes) — top priority, needed before drawing real conclusions
  from the single-seed numbers above
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
