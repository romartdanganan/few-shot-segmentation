"""
src/evaluate.py - evaluation protocol (Design Report, Section III-D):
methods are evaluated at k=1 and k=5 over several evaluation seeds, each
sampling a fixed set of episodes, reporting mean +/- standard deviation of
mIoU and F1. All methods see exactly the same episodes for a given seed,
so comparisons are paired.

--split val   is for tuning and model selection (validation classes).
--split test  is for the final numbers only; never tune on it.

Each checkpoint stores its own training settings (weighted, distance), so
they are read back automatically. Results, including every per-seed value
and the time per episode, are written as JSON.

Usage:
    python -m src.evaluate --data-root data/fewshot_data \
        --baseline-ckpt runs/baseline_k5_s0/best.pt \
        --prototype-ckpt runs/prototype_k5_s0/best.pt \
        --shots 1 5 --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import FSS1000Episodic, fss_collate
from src.models import (
    build_backbone,
    SegHead,
    baseline_query_logits,
    adapt_baseline,
    prototype_loss,
)
from src.metrics import binary_mask_metrics, RunningStats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Offset so the RNG used by test-time adaptation never shares a seed with
# the episode sampler.
TTA_SEED_OFFSET = 10_000


def load_model(ckpt_path, method):
    # Must match the architecture used when the checkpoint was saved, or
    # load_state_dict fails on a shape mismatch.
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    backbone = build_backbone().to(DEVICE)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()
    head = None
    if method == "baseline":
        c_out = backbone.config.hidden_sizes[-1]
        head = SegHead(c_out).to(DEVICE)
        head.load_state_dict(ckpt["head"])
        head.eval()
    train_args = ckpt.get("args", {})
    return backbone, head, train_args


def predict(backbone, head, method, s_imgs, s_masks, q_img, q_mask,
            weighted=False, distance="euclidean", adapt_steps=5, adapt_lr=1e-4):
    """One episode's query logits. Shared with scripts/analyze.py."""
    if method == "baseline":
        # adapt_baseline needs gradients internally, even when called from
        # inside a no_grad region.
        with torch.enable_grad():
            ab, ah = adapt_baseline(backbone, head, s_imgs, s_masks, lr=adapt_lr, steps=adapt_steps)
        with torch.no_grad():
            logits = baseline_query_logits(ab, ah, q_img)
        del ab, ah
        return logits
    with torch.no_grad():
        _, logits = prototype_loss(backbone, s_imgs, s_masks, q_img, q_mask,
                                   weighted=weighted, distance=distance)
    return logits


def run_eval(backbone, head, method, classes, data_root, k_shot, n_episodes, seed, img_size,
             weighted=False, distance="euclidean", adapt_steps=5, adapt_lr=1e-4):
    # augment=False: evaluate on the real images, not augmented variants.
    ds = FSS1000Episodic(
        data_root, classes, k_shot=k_shot, img_size=img_size,
        episodes_per_epoch=n_episodes, augment=False, seed=seed,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=fss_collate)
    stats = RunningStats()
    # Fixed RNG for the baseline's test-time adaptation (stochastic depth is
    # active while it fine-tunes), so repeated evaluations give identical numbers.
    torch.manual_seed(TTA_SEED_OFFSET + seed)
    elapsed = 0.0
    for s_imgs, s_masks, q_img, q_mask, cls in loader:
        s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
        q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = predict(backbone, head, method, s_imgs, s_masks, q_img, q_mask,
                         weighted, distance, adapt_steps, adapt_lr)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        stats.update(binary_mask_metrics(logits, q_mask))
    summary = stats.summary()
    summary["ms_per_episode"] = 1000.0 * elapsed / max(1, n_episodes)
    return summary


def evaluate_model(label, method, ckpt_path, args, classes):
    backbone, head, train_args = load_model(ckpt_path, method)
    weighted = bool(train_args.get("weighted", False)) or args.weighted
    distance = train_args.get("distance", "euclidean")
    out = {}
    for k in args.shots:
        per_seed = []
        for seed in args.seeds:
            s = run_eval(backbone, head, method, classes, args.data_root, k, args.episodes_per_seed,
                         seed, args.img_size, weighted, distance, args.adapt_steps, args.adapt_lr)
            per_seed.append(s)
            print(f"k={k} {label} seed={seed}: mIoU={s['mIoU'][0]:.4f}  F1={s['F1'][0]:.4f}  "
                  f"({s['ms_per_episode']:.1f} ms/episode)")
        miou = [s["mIoU"][0] for s in per_seed]
        f1 = [s["F1"][0] for s in per_seed]
        ms = [s["ms_per_episode"] for s in per_seed]
        key = f"k{k}_{label}"
        out[key] = {
            "method": method,
            "checkpoint": str(ckpt_path),
            "weighted": weighted,
            "distance": distance,
            "adapt_steps": args.adapt_steps if method == "baseline" else None,
            "adapt_lr": args.adapt_lr if method == "baseline" else None,
            "mIoU_mean": statistics.mean(miou),
            "mIoU_std": statistics.pstdev(miou) if len(miou) > 1 else 0.0,
            "F1_mean": statistics.mean(f1),
            "F1_std": statistics.pstdev(f1) if len(f1) > 1 else 0.0,
            "per_seed_mIoU": miou,
            "per_seed_F1": f1,
            "ms_per_episode": statistics.mean(ms),
            "seeds": args.seeds,
        }
        print(f"==> k={k} {label}: mIoU={out[key]['mIoU_mean']:.4f} +/- {out[key]['mIoU_std']:.4f}, "
              f"F1={out[key]['F1_mean']:.4f} +/- {out[key]['F1_std']:.4f}")
    del backbone, head
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-file", default="configs/class_splits.json")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--baseline-ckpt", default=None)
    ap.add_argument("--prototype-ckpt", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes-per-seed", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--weighted", action="store_true",
                    help="force the boundary-weighted variant (normally read from the checkpoint)")
    ap.add_argument("--adapt-steps", type=int, default=5, help="baseline: test-time adaptation steps")
    ap.add_argument("--adapt-lr", type=float, default=1e-4, help="baseline: test-time adaptation learning rate")
    ap.add_argument("--out", default="runs/eval_results.json")
    args = ap.parse_args()
    if not (args.baseline_ckpt or args.prototype_ckpt):
        ap.error("give --baseline-ckpt and/or --prototype-ckpt")

    with open(args.splits_file) as f:
        splits = json.load(f)
    # Same classes and same seeds for every method: a fair, paired
    # comparison, not just "whoever got easier episodes".
    classes = splits[args.split]
    print(f"Evaluating on {len(classes)} {args.split} classes.")

    results = {"split": args.split, "episodes_per_seed": args.episodes_per_seed,
               "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"}
    if args.baseline_ckpt:
        results.update(evaluate_model("baseline", "baseline", args.baseline_ckpt, args, classes))
    if args.prototype_ckpt:
        results.update(evaluate_model("prototype", "prototype", args.prototype_ckpt, args, classes))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
