# Parameter tuning (validation classes, training seed 0, k=5)

Selection metric: mean validation mIoU over 3 evaluation seeds x 50 episodes.

## baseline: learning_rate (chosen: 0.0001)

| value | val mIoU | ± std | val F1 |
|---|---|---|---|
| 3e-05 | 0.8364 | 0.0046 | 0.8398 |
| 0.0001 | 0.8460 | 0.0029 | 0.8523 |
| 0.0003 | 0.8179 | 0.0025 | 0.8192 |

## baseline: epochs (chosen: 10)

| value | val mIoU | ± std | val F1 |
|---|---|---|---|
| 10 | 0.8460 | 0.0029 | 0.8523 |
| 20 | 0.8442 | 0.0038 | 0.8503 |

## baseline: tta_steps (chosen: 10)

| value | val mIoU | ± std | val F1 |
|---|---|---|---|
| 0 | 0.8122 | 0.0113 | 0.8145 |
| 1 | 0.8198 | 0.0072 | 0.8126 |
| 5 | 0.8460 | 0.0029 | 0.8523 |
| 10 | 0.8505 | 0.0029 | 0.8587 |

## baseline: tta_lr (chosen: 0.0001)

| value | val mIoU | ± std | val F1 |
|---|---|---|---|
| 3e-05 | 0.8412 | 0.0048 | 0.8451 |
| 0.0001 | 0.8505 | 0.0029 | 0.8587 |
| 0.0003 | 0.8395 | 0.0037 | 0.8467 |

## prototype: learning_rate (chosen: 0.0001)

| value | val mIoU | ± std | val F1 |
|---|---|---|---|
| 3e-05 | 0.7982 | 0.0045 | 0.8059 |
| 0.0001 | 0.8176 | 0.0070 | 0.8267 |
| 0.0003 | 0.7581 | 0.0158 | 0.7628 |

## prototype: epochs (chosen: 20)

| value | val mIoU | ± std | val F1 |
|---|---|---|---|
| 10 | 0.8176 | 0.0070 | 0.8267 |
| 20 | 0.8180 | 0.0099 | 0.8261 |

## prototype: distance (chosen: euclidean)

| value | val mIoU | ± std | val F1 |
|---|---|---|---|
| euclidean | 0.8180 | 0.0099 | 0.8261 |
| cosine | 0.8171 | 0.0014 | 0.8219 |

Chosen settings: `{"baseline": {"lr": 0.0001, "epochs": 10, "tta": [10, 0.0001]}, "prototype": {"lr": 0.0001, "epochs": 20, "distance": "euclidean"}}`