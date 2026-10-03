# Paired tests across training seeds (A minus B)

Paired t-test (primary) and Wilcoxon signed-rank (check, uncorrected); Holm correction within each metric's family of comparisons; alpha = 0.05; Cohen's d_z = mean difference / sd of differences.

| metric | k | A | B | n | diff (A-B) | t | p (t) | p Holm | p Wilcoxon | d_z |
|---|---|---|---|---|---|---|---|---|---|---|
| mIoU | 1 | Fine-tune | Proto | 10 | +0.0411 | 13.01 | 3.85e-07 | 7.71e-07 | 0.00195 | +4.11 |
| mIoU | 5 | Fine-tune | Proto | 10 | +0.0173 | 20.18 | 8.4e-09 | 2.52e-08 | 0.00195 | +6.38 |
| mIoU | 5 | Proto | Proto-BW | 10 | +0.0040 | 2.95 | 0.0162 | 0.0162 | 0.00977 | +0.93 |
| mIoU | 5 | Fine-tune | Fine-tune, no TTA | 10 | +0.0283 | 22.81 | 2.84e-09 | 1.14e-08 | 0.00195 | +7.21 |
| F1 | 1 | Fine-tune | Proto | 10 | +0.0587 | 12.57 | 5.18e-07 | 1.04e-06 | 0.00195 | +3.98 |
| F1 | 5 | Fine-tune | Proto | 10 | +0.0138 | 16.52 | 4.87e-08 | 1.46e-07 | 0.00195 | +5.22 |
| F1 | 5 | Proto | Proto-BW | 10 | +0.0043 | 3.45 | 0.00724 | 0.00724 | 0.00977 | +1.09 |
| F1 | 5 | Fine-tune | Fine-tune, no TTA | 10 | +0.0240 | 19.14 | 1.34e-08 | 5.36e-08 | 0.00195 | +6.05 |