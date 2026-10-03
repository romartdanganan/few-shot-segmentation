# Computational cost

| Condition | train time (min) | best epoch (per seed) | peak GPU MB | total params | ms/episode k=1 | ms/episode k=5 |
|---|---|---|---|---|---|---|
| Fine-tune | 2.9 ± 0.2 | [9, 7, 8, 9, 8, 10, 9, 9, 9, 6] | 334 | 3,319,906 | 3921 | 296 |
| Proto | 8.3 ± 1.0 | [20, 14, 14, 17, 14, 14, 16, 18, 15, 12] | 251 | 3,319,392 | 16 | 16 |
| Proto-BW | 8.1 ± 0.9 | [16, 14, 14, 17, 14, 20, 18, 12, 10, 15] | 251 | 3,319,392 | 16 | 16 |
| Fine-tune, no TTA | (same model) | | | | 26 | 30 |

Hardware: NVIDIA GeForce RTX 3070