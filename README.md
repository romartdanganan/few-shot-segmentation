# Few-Shot Semantic Segmentation with Prototype-Based Loss

Few-shot semantic segmentation on [FSS-1000](https://github.com/HKUSTCV/FSS-1000), comparing standard
fine-tuning against an episodic, prototype-based training objective on a shared
[SegFormer (MiT-B0)](https://arxiv.org/abs/2105.15203) backbone.

AIML339 capstone project, Victoria University of Wellington. The class-level data splits, episodic
sampler, both training objectives, the boundary-weighted variant, the evaluation protocol and the
analysis scripts are implemented in this repository; the pretrained SegFormer weights and the
libraries in `requirements.txt` are reused unmodified.

## The idea

Segmentation models normally learn from thousands of pixel-level masks per class. This project asks:
**can a model segment a brand-new class from just 1 or 5 labelled examples**, and does
prototype-based episodic training do this better than standard fine-tuning?

All methods share the same backbone, data, optimiser and evaluation episodes, so differences come
from the training objective:

| | Fine-tune (baseline) | Proto (prototype-based) |
|---|---|---|
| Backbone | SegFormer MiT-B0 encoder (ImageNet-1K, then ADE20K) | same, shared initialisation |
| Training | 1x1 conv head, cross-entropy on the support images of each training episode | masked average pooling of support features into foreground/background prototypes; query pixels scored by distance to them; cross-entropy on the query |
| New class at test time | copy the model, a few gradient steps on the support set, predict the query | build prototypes from the support set, one forward pass |
| Ablation | | Proto-BW: prototypes down-weight ambiguous mask-boundary locations by \|2m - 1\| |

## Project status

- [x] Feasibility pilot (`scripts/pilot_test.py`): pipeline runs and fits easily in 8 GB of VRAM (RTX 3070)
- [x] Data pipeline with class-level train/val/test splits (no class in more than one role)
- [x] Fine-tune, Proto and Proto-BW training, plus the evaluation protocol (k=1/k=5, several seeds)
- [x] Two silent bugs fixed (prototype reshape, frozen training episodes), covered by `scripts/sanity_check.py` (9 tests)
- [x] Seed-0 models retrained after both fixes; early single-seed analysis kept in `analysis/` and `results/`
- [x] Full protocol pipeline (`scripts/run_experiments.py`) and stats/figures script, smoke-tested
- [x] Full protocol run: tuning on validation classes, 30 training seeds, test evaluation (`experiments/`)
- [x] Tables, statistical tests and figures from the full run (`experiments/report/`)
- [x] Final report updated with the full-run results
- [ ] Final presentation updated with the full-run results

## Repository structure

```
.
├── src/
│   ├── dataset.py     # FSS-1000 episodic dataset + class-level splitting
│   ├── models.py      # SegFormer encoder, baseline head + test-time adaptation, prototype loss
│   ├── train.py       # training loop for either method; writes history.json per run
│   ├── evaluate.py    # val/test evaluation over seeds; per-seed scores and timing to JSON
│   └── metrics.py     # mIoU (mean of foreground and background IoU) and foreground F1
├── scripts/
│   ├── run_experiments.py  # the full protocol: tuning, multi-seed training, test evaluation
│   ├── stats_and_figures.py# every table, statistical test and figure in the report
│   ├── near_duplicate_check.py # effect of near-duplicate test classes (same episodes)
│   ├── analyze.py          # per-class results and qualitative examples
│   ├── sanity_check.py     # synthetic-input tests of the core logic (no GPU or data needed)
│   └── pilot_test.py       # original feasibility pilot (synthetic data, memory check)
├── configs/class_splits.json   # the fixed 700/100/200 class split (seed 0)
├── experiments/        # JSON logs of every run behind the final report (checkpoints not committed)
├── results/, analysis/, runs/  # earlier single-seed runs and their TensorBoard logs, kept for reference
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
python -m scripts.sanity_check   # should print "9/9 checks passed"
```

## Data preparation

Download [FSS-1000](https://github.com/HKUSTCV/FSS-1000) (also on
[Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/fss1000-a-1000-class-fewshot-segmentation)).
It has 1000 classes with 10 image/mask pairs each (224x224). It is not included in this repository: its
images were collected from web image searches and no licence is stated, so use it for research only and
cite the FSS-1000 paper.
Unzip it so each class is a folder of numbered pairs:

```
data/
└── fewshot_data/
    ├── abacus/
    │   ├── 1.jpg
    │   ├── 1.png
    │   └── ...
    └── ...
```

The class-level split in `configs/class_splits.json` (700 train / 100 validation / 200 test classes,
created once with seed 0) is used by every script, so no class appears in more than one role.
Validation classes are used for tuning and checkpoint selection only; test classes are used only
for the final numbers.

## Reproducing the report

One resumable command runs the whole protocol (re-run the same command after a crash or reboot;
finished runs are skipped):

```bash
python -m scripts.run_experiments --data-root data/fewshot_data --smoke   # quick pipeline check
python -m scripts.run_experiments --data-root data/fewshot_data --n-seeds 30   # full protocol (as reported)
python -m scripts.stats_and_figures                                       # tables, tests, figures
python -m scripts.near_duplicate_check --data-root data/fewshot_data      # near-duplicate class check
```

What the full protocol does (all settings are logged to `experiments/state.json`):

1. **Tuning, one aspect at a time, on the validation classes** (training seed 0, 5-shot, selection by
   mean validation mIoU over 3 x 50 episodes). Starting points follow the literature: AdamW as in
   SegFormer, batch size of one episode as in PANet.
   - learning rate {3e-5, 1e-4, 3e-4} for each method
   - training epochs {10, 20} (200 episodes each)
   - Fine-tune: test-time adaptation steps {0, 1, 5, 10}, then its learning rate {3e-5, 1e-4, 3e-4}
   - Proto: distance {squared Euclidean / sqrt(C), cosine x 20 as in PANet}
2. **Final training** with the chosen settings for Fine-tune, Proto and Proto-BW over 30 training seeds (0 to 29).
3. **Test evaluation** of every final model at k=1 and k=5 over 10 evaluation seeds x 50 episodes, with
   identical episodes for every model; plus Fine-tune without test-time adaptation as a diagnostic.
4. **Per-class analysis and qualitative examples** for the seed-0 models.

Which files correspond to the report:

| Report item | File |
|---|---|
| Parameter-tuning tables | `experiments/report/tuning_tables.md`, `experiments/state.json` |
| Main results table | `experiments/report/final_table.md` (from `experiments/test/*.json`) |
| Statistical tests | `experiments/report/stats.md` / `stats.json` |
| Cost table | `experiments/report/cost_table.md` (from each run's `history.json`) |
| Validation curves / run distributions | `experiments/report/fig_val.png`, `fig_box.png` |
| Per-class results, qualitative examples | `experiments/analysis/*_k5/` |
| Friedman test | `experiments/report/friedman.json` |
| Near-duplicate class check | `experiments/near_duplicates/summary.json` |
| Every training run (config, per-epoch loss and validation, time, memory) | `experiments/train/*/history.json` |

Individual scripts can also be run directly, e.g.:

```bash
python -m src.train --method prototype --data-root data/fewshot_data --lr 1e-4 --epochs 10 --seed 0
python -m src.evaluate --data-root data/fewshot_data --split val --prototype-ckpt experiments/train/prototype_euclidean_lr0.0001_e20_s0/best.pt --shots 5 --seeds 0 1 2
tensorboard --logdir experiments/train
```

## Reproducibility and randomness

- Training episodes depend on the training seed and the epoch; validation and test episodes depend
  only on the evaluation seed, so every model sees the same validation and test episodes.
- The baseline's test-time adaptation runs with stochastic depth active, so its random generator is
  seeded per evaluation seed; repeated evaluations give identical numbers.
- Seeds are fixed integers (never clock time) and are recorded in every JSON file.

## Development history

Two silent defects were found and fixed during the project (see the commit history):
the prototype pooling reshaped a (batch, k, C, h, w) tensor without first moving the channel axis
(caught by `scripts/sanity_check.py`), and training episodes were identical in every epoch
(found in a code review; now covered by a regression test). All results in `experiments/` come from
code after both fixes. Results in `results/` and `analysis/` are earlier single-seed runs kept for
reference.

## References

- Xie et al., ["SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"](https://arxiv.org/abs/2105.15203), NeurIPS 2021
- Wang et al., ["PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment"](https://arxiv.org/abs/1908.06391), ICCV 2019
- Snell et al., ["Prototypical Networks for Few-Shot Learning"](https://arxiv.org/abs/1703.05175), NeurIPS 2017
- Li et al., ["FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation"](https://arxiv.org/abs/1907.12347), CVPR 2020

## License

MIT, see [LICENSE](LICENSE).
