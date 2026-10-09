# Computational cost

| Condition | train time (min) | best epoch (per seed) | peak GPU MB | total params | median ms/episode k=1 | median ms/episode k=5 |
|---|---|---|---|---|---|---|
| Fine-tune | 2.5 ± 0.2 | [9, 7, 8, 9, 8, 10, 9, 9, 9, 6, 9, 8, 8, 10, 8, 10, 9, 9, 10, 10, 10, 8, 9, 8, 10, 7, 10, 10, 6, 5] | 334 | 3,319,906 | 305 | 311 |
| Proto | 7.5 ± 0.9 | [20, 14, 14, 17, 14, 14, 16, 18, 15, 12, 13, 16, 17, 15, 17, 18, 20, 16, 19, 17, 18, 18, 17, 17, 16, 10, 19, 16, 15, 10] | 251 | 3,319,392 | 16 | 17 |
| Proto-BW | 7.4 ± 0.8 | [16, 14, 14, 17, 14, 20, 18, 12, 10, 15, 13, 17, 19, 17, 15, 18, 11, 9, 11, 13, 19, 18, 8, 19, 17, 12, 20, 16, 15, 17] | 251 | 3,319,392 | 17 | 17 |
| Fine-tune, no TTA | (same model) | | | | 28 | 32 |

Hardware: NVIDIA GeForce RTX 3070
Test times are medians over training seeds, because a few runs were slowed by the computer sleeping.