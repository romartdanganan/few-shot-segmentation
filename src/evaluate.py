"""
src/evaluate.py — final evaluation protocol (Design Report, Section
III-D): both methods evaluated at k=1 and k=5, over multiple episodes and
seeds, reporting mean +/- standard deviation of mIoU and F1. Also does
the paired comparison: baseline and prototype use the *same* sampled
episodes at a given seed, so the comparison is fair.

Usage:
    python -m src.evaluate --data-root data/FSS-1000 \
        --baseline-ckpt runs/baseline_k5_s0/best.pt \
        --prototype-ckpt runs/prototype_k5_s0/best.pt \
        --shots 1 5 --seeds 0 1 2
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import FSS1000Episodic, class_level_split, discover_classes, fss_collate
from src.models import build_backbone, SegHead, baseline_loss, prototype_loss
from src.metrics import binary_mask_metrics, RunningStats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone_and_head(ckpt_path, method):
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
    return backbone, head


@torch.no_grad()
def run_eval(backbone, head, method, weighted, data_root, test_classes, k_shot, n_episodes, seed, img_size):
    ds = FSS1000Episodic(
        data_root, test_classes, k_shot=k_shot, img_size=img_size,
        episodes_per_epoch=n_episodes, augment=False, seed=seed,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=fss_collate)
    stats = RunningStats()
    for s_imgs, s_masks, q_img, q_mask, cls in loader:
        s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
        q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)
        if method == "baseline":
            _, logits = baseline_loss(backbone, head, s_imgs, s_masks)
        else:
            _, logits = prototype_loss(backbone, s_imgs, s_masks, q_img, q_mask, weighted=weighted)
        stats.update(binary_mask_metrics(logits, q_mask))
    return stats.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-file", default="configs/class_splits.json")
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--prototype-ckpt", required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes-per-seed", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--out", default="runs/eval_results.json")
    args = ap.parse_args()

    with open(args.splits_file) as f:
        splits = json.load(f)
    test_classes = splits["test"]
    print(f"Evaluating on {len(test_classes)} held-out test classes.")

    baseline_backbone, baseline_head = load_backbone_and_head(args.baseline_ckpt, "baseline")
    proto_backbone, _ = load_backbone_and_head(args.prototype_ckpt, "prototype")

    results = {}
    for k in args.shots:
        for method, backbone, head in [
            ("baseline", baseline_backbone, baseline_head),
            ("prototype", proto_backbone, None),
        ]:
            per_seed = []
            for seed in args.seeds:
                summary = run_eval(
                    backbone, head, method, args.weighted, args.data_root, test_classes,
                    k_shot=k, n_episodes=args.episodes_per_seed, seed=seed, img_size=args.img_size,
                )
                per_seed.append(summary)
                print(f"k={k} method={method} seed={seed}: "
                      f"mIoU={summary['mIoU'][0]:.4f}  F1={summary['F1'][0]:.4f}")

            miou_vals = [s["mIoU"][0] for s in per_seed]
            f1_vals = [s["F1"][0] for s in per_seed]
            import statistics
            key = f"k{k}_{method}"
            results[key] = {
                "mIoU_mean": statistics.mean(miou_vals),
                "mIoU_std": statistics.pstdev(miou_vals) if len(miou_vals) > 1 else 0.0,
                "F1_mean": statistics.mean(f1_vals),
                "F1_std": statistics.pstdev(f1_vals) if len(f1_vals) > 1 else 0.0,
                "seeds": args.seeds,
            }
            print(f"==> k={k} {method}: mIoU={results[key]['mIoU_mean']:.4f} "
                  f"+/- {results[key]['mIoU_std']:.4f}, "
                  f"F1={results[key]['F1_mean']:.4f} +/- {results[key]['F1_std']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
