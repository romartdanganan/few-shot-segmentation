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