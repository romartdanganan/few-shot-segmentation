"""
scripts/analyze.py — per-class error analysis and qualitative examples
(Design Report Section IV: per-class error breakdown, qualitative mask
comparison).

Usage:
    python -m scripts.analyze --data-root data/fewshot_data \
        --checkpoint runs/baseline_k5_s0/best.pt --method baseline \
        --k-shot 5 --out-dir analysis/baseline_k5

    python -m scripts.analyze --data-root data/fewshot_data \
        --checkpoint runs/prototype_k5_s0/best.pt --method prototype \
        --k-shot 5 --out-dir analysis/prototype_k5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import FSS1000Episodic, fss_collate
from src.models import build_backbone, SegHead, baseline_query_logits, adapt_baseline, prototype_loss
from src.metrics import binary_mask_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(img_t):
    """Undo dataset.py's normalization, back to a displayable RGB image."""
    img = img_t.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()


def mask_to_rgb(mask, color):
    """0/1 mask -> colored image, ready to blend onto the base photo."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask.cpu().numpy() > 0.5] = color
    return rgb


def overlay(img, mask_rgb, alpha=0.45):
    out = img.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return out.astype(np.uint8)


def save_qualitative(query_img, gt_mask, pred_mask, out_path):
    """Save [original | ground truth | prediction] side by side."""
    base = denormalize(query_img[0])
    gt_rgb = overlay(base, mask_to_rgb(gt_mask[0], (0, 255, 0)))
    pred_rgb = overlay(base, mask_to_rgb(pred_mask[0], (255, 0, 0)))
    Image.fromarray(np.concatenate([base, gt_rgb, pred_rgb], axis=1)).save(out_path)


def load_model(ckpt_path, method):
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


def predict(backbone, head, method, weighted, s_imgs, s_masks, q_img, q_mask):
    """Same prediction logic as evaluate.py's run_eval, reused here."""
    if method == "baseline":
        with torch.enable_grad():
            ab, ah = adapt_baseline(backbone, head, s_imgs, s_masks)
        logits = baseline_query_logits(ab, ah, q_img)
        del ab, ah
    else:
        with torch.no_grad():
            _, logits = prototype_loss(backbone, s_imgs, s_masks, q_img, q_mask, weighted=weighted)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-file", default="configs/class_splits.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", choices=["baseline", "prototype"], required=True)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--episodes-per-class", type=int, default=3)
    ap.add_argument("--n-qualitative", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    with open(args.splits_file) as f:
        splits = json.load(f)
    test_classes = splits["test"]

    backbone, head = load_model(args.checkpoint, args.method)

    # ---- per-class breakdown ----
    # One class at a time so results/mIoU can be reported per class,
    # not just as one averaged number across all of them.
    per_class = {}
    for i, cls in enumerate(test_classes):
        ds = FSS1000Episodic(
            args.data_root, [cls], k_shot=args.k_shot, img_size=args.img_size,
            episodes_per_epoch=args.episodes_per_class, augment=False,
            seed=args.seed + i,
        )
        ious, f1s = [], []
        for j in range(len(ds)):
            s_imgs, s_masks, q_img, q_mask, _ = fss_collate([ds[j]])
            s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
            q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)
            logits = predict(backbone, head, args.method, args.weighted, s_imgs, s_masks, q_img, q_mask)
            m = binary_mask_metrics(logits, q_mask)
            ious.append(m["mIoU"])
            f1s.append(m["F1"])
        per_class[cls] = {"mIoU": sum(ious) / len(ious), "F1": sum(f1s) / len(f1s)}
        print(f"[{i+1}/{len(test_classes)}] {cls}: mIoU={per_class[cls]['mIoU']:.4f}")

    with open(out_dir / "per_class_results.json", "w") as f:
        json.dump(per_class, f, indent=2)

    ranked = sorted(per_class.items(), key=lambda kv: kv[1]["mIoU"])
    print("\nWorst 10 classes:")
    for cls, m in ranked[:10]:
        print(f"  {cls}: mIoU={m['mIoU']:.4f}")
    print("\nBest 10 classes:")
    for cls, m in ranked[-10:]:
        print(f"  {cls}: mIoU={m['mIoU']:.4f}")

    # ---- qualitative examples ----
    # A handful of side-by-side comparisons: input | ground truth | prediction.
    qual_ds = FSS1000Episodic(
        args.data_root, test_classes, k_shot=args.k_shot, img_size=args.img_size,
        episodes_per_epoch=args.n_qualitative, augment=False, seed=args.seed + 9999,
    )
    for i in range(len(qual_ds)):
        s_imgs, s_masks, q_img, q_mask, cls = fss_collate([qual_ds[i]])
        s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
        q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)
        logits = predict(backbone, head, args.method, args.weighted, s_imgs, s_masks, q_img, q_mask)
        pred_mask = logits.argmax(dim=1)
        save_qualitative(q_img, q_mask, pred_mask, qual_dir / f"{i:02d}_{cls[0]}.png")

    print(f"\nSaved per-class results and {args.n_qualitative} qualitative examples to {out_dir}")


if __name__ == "__main__":
    main()
