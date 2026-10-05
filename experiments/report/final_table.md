# Test results (200 test classes)

Mean ± std over training seeds; each seed's score is the mean over the evaluation seeds.

| Condition | k | n seeds | mIoU | F1 | mean within-model eval-seed std (mIoU) |
|---|---|---|---|---|---|
| Fine-tune | 1 | 30 | 0.808 ± 0.004 | 0.799 ± 0.005 | 0.015 |
| Fine-tune, no TTA | 1 | 30 | 0.802 ± 0.007 | 0.800 ± 0.010 | 0.014 |
| Proto | 1 | 30 | 0.768 ± 0.010 | 0.743 ± 0.015 | 0.019 |
| Proto-BW | 1 | 30 | 0.763 ± 0.009 | 0.739 ± 0.015 | 0.018 |
| Fine-tune | 5 | 30 | 0.830 ± 0.002 | 0.824 ± 0.002 | 0.014 |
| Fine-tune, no TTA | 5 | 30 | 0.800 ± 0.007 | 0.797 ± 0.010 | 0.017 |
| Proto | 5 | 30 | 0.812 ± 0.003 | 0.811 ± 0.004 | 0.014 |
| Proto-BW | 5 | 30 | 0.809 ± 0.003 | 0.806 ± 0.003 | 0.015 |

std = sample standard deviation across training seeds (ddof=1).