# Paired tests across training seeds (A minus B)

Paired t-test (primary) and Wilcoxon signed-rank (check, uncorrected); Holm correction within each metric's family of comparisons; alpha = 0.05; Cohen's d_z = mean difference / sd of differences.

| metric | k | A | B | n | diff (A-B) | t | p (t) | p Holm | p Wilcoxon | d_z |
|---|---|---|---|---|---|---|---|---|---|---|
| mIoU | 1 | Fine-tune | Proto | 30 | +0.0403 | 21.22 | 3.28e-19 | 6.56e-19 | 1.86e-09 | +3.87 |
| mIoU | 5 | Fine-tune | Proto | 30 | +0.0179 | 25.68 | 1.7e-21 | 5.09e-21 | 1.86e-09 | +4.69 |
| mIoU | 5 | Proto | Proto-BW | 30 | +0.0038 | 5.43 | 7.69e-06 | 7.69e-06 | 5.14e-06 | +0.99 |
| mIoU | 5 | Fine-tune | Fine-tune, no TTA | 30 | +0.0301 | 26.30 | 8.76e-22 | 3.51e-21 | 1.86e-09 | +4.80 |
| F1 | 1 | Fine-tune | Proto | 30 | +0.0559 | 19.43 | 3.58e-18 | 1.43e-17 | 1.86e-09 | +3.55 |
| F1 | 5 | Fine-tune | Proto | 30 | +0.0137 | 17.50 | 5.95e-17 | 1.78e-16 | 1.86e-09 | +3.19 |
| F1 | 5 | Proto | Proto-BW | 30 | +0.0045 | 6.42 | 5.02e-07 | 5.02e-07 | 2.76e-06 | +1.17 |
| F1 | 5 | Fine-tune | Fine-tune, no TTA | 30 | +0.0273 | 15.60 | 1.22e-15 | 2.44e-15 | 1.86e-09 | +2.85 |

## Robustness: same tests on even and odd training seeds (mIoU, Holm within each half)

| seeds | k | A | B | n | diff (A-B) | p Holm | p Wilcoxon | d_z |
|---|---|---|---|---|---|---|---|---|
| even | 1 | Fine-tune | Proto | 15 | +0.0425 | 8.99e-10 | 6.1e-05 | +4.01 |
| even | 5 | Fine-tune | Proto | 15 | +0.0169 | 2.19e-10 | 6.1e-05 | +4.58 |
| even | 5 | Proto | Proto-BW | 15 | +0.0040 | 0.000475 | 0.00061 | +1.17 |
| even | 5 | Fine-tune | Fine-tune, no TTA | 15 | +0.0302 | 8.99e-10 | 6.1e-05 | +4.03 |
| odd | 1 | Fine-tune | Proto | 15 | +0.0381 | 1.38e-09 | 6.1e-05 | +3.78 |
| odd | 5 | Fine-tune | Proto | 15 | +0.0190 | 4.58e-11 | 6.1e-05 | +5.03 |
| odd | 5 | Proto | Proto-BW | 15 | +0.0036 | 0.00619 | 0.00836 | +0.83 |
| odd | 5 | Fine-tune | Fine-tune, no TTA | 15 | +0.0301 | 6.2e-12 | 6.1e-05 | +5.95 |

Unplanned check, Proto minus Proto-BW at k=1 (mIoU, n=30): diff +0.0042, paired t-test p = 0.0652 (uncorrected), Wilcoxon p = 0.0667, d_z = +0.35.