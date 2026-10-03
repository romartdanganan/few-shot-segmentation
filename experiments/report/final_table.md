# Test results (200 test classes)

Mean ± std over training seeds; each seed's score is the mean over the evaluation seeds.

| Condition | k | n seeds | mIoU | F1 | mean within-model eval-seed std (mIoU) |
|---|---|---|---|---|---|
| Fine-tune | 1 | 10 | 0.808 ± 0.003 | 0.798 ± 0.004 | 0.015 |
| Fine-tune, no TTA | 1 | 10 | 0.803 ± 0.006 | 0.803 ± 0.005 | 0.015 |
| Proto | 1 | 10 | 0.766 ± 0.010 | 0.740 ± 0.015 | 0.020 |
| Proto-BW | 1 | 10 | 0.763 ± 0.012 | 0.738 ± 0.019 | 0.020 |
| Fine-tune | 5 | 10 | 0.830 ± 0.003 | 0.824 ± 0.003 | 0.014 |
| Fine-tune, no TTA | 5 | 10 | 0.802 ± 0.006 | 0.800 ± 0.005 | 0.017 |
| Proto | 5 | 10 | 0.813 ± 0.003 | 0.810 ± 0.003 | 0.014 |
| Proto-BW | 5 | 10 | 0.809 ± 0.004 | 0.806 ± 0.003 | 0.016 |

std = sample standard deviation across training seeds (ddof=1).