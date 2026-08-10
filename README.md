# Few-Shot Semantic Segmentation with Prototype-Based Loss

Few-shot semantic segmentation on [FSS-1000](https://github.com/HKUSTCV/FSS-1000), comparing standard
fine-tuning against an episodic, prototype-based training objective on a shared
[SegFormer (MiT-B0)](https://arxiv.org/abs/2105.15203) backbone.

This started as a university project (AIML339, Victoria University of Wellington) and is developed
here as a from-scratch implementation with its own class-level data splits, training loop, and
evaluation protocol.

## The idea

Semantic segmentation models need a lot of pixel-level labels. New object classes almost never have
that much data available. This project asks: **can a model learn to segment a brand-new class from
just 1–5 labelled examples**, using prototype-based episodic training instead of standard fine-tuning?

Two methods are trained on the *same* backbone, *same* data budget, and *same* evaluation protocol, so
any difference in results comes from the training objective itself:

| | Method A — Baseline | Method B — Prototype-based |
|---|---|---|
| Backbone | SegFormer MiT-B0 (pretrained, ADE20K) | SegFormer MiT-B0 (shared) |
| Training | Standard cross-entropy fine-tuning on the k-shot support set | Episodic: masked-average-pool a class prototype from the support set, classify query pixels by distance to it |
| Ablation | — | Distance-weighted variant that down-weights support pixels near the mask boundary |

Both are evaluated at **k=1** and **k=5** shots, across multiple episodes and random seeds, reporting
mean ± standard deviation of **mIoU** and **F1-score**.

## Project status

- [x] Feasibility pilot (`scripts/pilot_test.py`) — confirmed the training pipeline is correct and
      fits comfortably in 8 GB of VRAM on a GTX/RTX 3070 (~200 MB peak usage at 256×256)
- [x] Full data pipeline with class-level train/val/test splits (no leakage between seen and novel classes)
- [x] Baseline and prototype-based training scripts
- [x] Evaluation protocol (k=1/k=5, multi-seed, mean ± std)
- [ ] Full training run and final results (in progress)
- [ ] Distance-weighted ablation results
- [ ] Qualitative examples and per-class error analysis

## Repository structure

```
.
├── src/
│   ├── dataset.py     # FSS-1000 episodic dataset + class-level splitting
│   ├── models.py       # SegFormer backbone, baseline head, prototype loss (Eq. 1–2)
│   ├── train.py        # training loop for either method
│   ├── evaluate.py     # k=1/k=5 evaluation across seeds, mean ± std
│   └── metrics.py       # mIoU / F1
├── scripts/
│   └── pilot_test.py   # the original feasibility pilot (synthetic data, GPU/memory check)
├── configs/            # class-level train/val/test splits get written here
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download [FSS-1000](https://github.com/HKUSTCV/FSS-1000) (also mirrored on
[Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/fss1000-a-1000-class-fewshot-segmentation))
and unzip it so each class has its own folder of numbered image/mask pairs, e.g.:

This project uses FSS-1000, containing 1000 object classes with 10 annotated image-mask pairs per class.

The dataset is not included in this repository.

Expected local layout:

data/
└── fewshot_data/
    ├── abacus/
    │   ├── 1.jpg
    │   ├── 1.png
    │   ├── 2.jpg
    │   ├── 2.png
    │   └── ...
    ├── accordion/
    └── ...

The loader uses a class-level split of:

- 700 training classes
- 100 validation classes
- 200 test classes

There is no class overlap between splits.

Both 1-shot and 5-shot episodes consist of a k-shot support set and one query image.

## Usage

**Train the baseline:**
```bash
python -m src.train --data-root data/FSS-1000 --method baseline --img-size 256
```

**Train the prototype-based method:**
```bash
python -m src.train --data-root data/FSS-1000 --method prototype --img-size 256
```

**Train the distance-weighted ablation:**
```bash
python -m src.train --data-root data/FSS-1000 --method prototype --weighted --img-size 256
```

**Evaluate both methods at k=1 and k=5, across 3 seeds:**
```bash
python -m src.evaluate --data-root data/FSS-1000 \
    --baseline-ckpt runs/baseline_k5_s0/best.pt \
    --prototype-ckpt runs/prototype_k5_s0/best.pt \
    --shots 1 5 --seeds 0 1 2
```

Training progress can be monitored with TensorBoard:
```bash
tensorboard --logdir runs
```

## Current development status

Real FSS-1000 integration has been verified with:

- 1000 detected classes
- 700/100/200 train/validation/test class split
- 1-shot and 5-shot episodic sampling
- CUDA training on an RTX 3070
- standard fine-tuning baseline
- prototype-based episodic training
- distance-weighted prototype ablation

Short smoke tests have been completed on all three training conditions.


## References

- Xie et al., ["SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"](https://arxiv.org/abs/2105.15203), NeurIPS 2021
- Wang et al., ["PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment"](https://arxiv.org/abs/1908.06391), ICCV 2019
- Snell et al., ["Prototypical Networks for Few-Shot Learning"](https://arxiv.org/abs/1703.05175), NeurIPS 2017
- Li et al., ["FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation"](https://arxiv.org/abs/1907.12347), CVPR 2020

## License

MIT — see [LICENSE](LICENSE).
