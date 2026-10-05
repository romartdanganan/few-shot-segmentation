"""
scripts/near_duplicate_check.py - leakage check for near-duplicate classes.

FSS-1000 has a few numbered sibling classes (e.g. earphone1 / earphone2). If one
sibling is a training class and the other a test class, the test class is not
truly novel. This script replays the main test episodes for the final Fine-tune
and Proto models (same seeds, same settings, same episodes), records the class
and score of every episode, and compares the mean with and without the episodes
drawn from those test classes. Removing classes from the list instead would
change every sampled episode, so it would not isolate their effect.

Usage (after scripts.run_experiments has finished):
    python -m scripts.near_duplicate_check --data-root data/fewshot_data
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import FSS1000Episodic, fss_collate
from src.evaluate import load_model, predict, TTA_SEED_OFFSET, DEVICE
from src.metrics import binary_mask_metrics


def find_near_duplicates(splits):
    """Test classes whose numbered sibling (same name, different trailing digits) is a training class."""
    base = lambda c: re.sub(r"\d+$", "", c)
    train_bases = {}
    for c in splits["train"]:
        train_bases.setdefault(base(c), []).append(c)
    pairs = []
    for c in splits["test"]:
        siblings = [t for t in train_bases.get(base(c), []) if t != c]
        if siblings and (re.search(r"\d+$", c) or any(re.search(r"\d+$", t) for t in siblings)):
            pairs.append((c, siblings))
    return pairs


def episode_scores(ckpt, method, classes, args, k, seed, tta):
    """Same loop as src.evaluate.run_eval, but keeps every episode's class and scores."""
    backbone, head, train_args = load_model(ckpt, method)
    ds = FSS1000Episodic(args.data_root, classes, k_shot=k, img_size=args.img_size,
                         episodes_per_epoch=args.episodes, augment=False, seed=seed)
    loader = DataLoader(ds, batch_size=1, collate_fn=fss_collate)
    torch.manual_seed(TTA_SEED_OFFSET + seed)
    rows = []
    for s_imgs, s_masks, q_img, q_mask, cls in loader:
        s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
        q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)
        logits = predict(backbone, head, method, s_imgs, s_masks, q_img, q_mask,
                         bool(train_args.get("weighted", False)), train_args.get("distance", "euclidean"),
                         tta[0], tta[1])
        m = binary_mask_metrics(logits, q_mask)
        rows.append({"class": cls[0], "mIoU": m["mIoU"], "F1": m["F1"]})
    del backbone, head
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-file", default="configs/class_splits.json")
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--n-seeds", type=int, default=10, help="training seeds to check (0..n-1)")
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--exclude", nargs="*", default=None, help="override the automatic list of test classes")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / "near_duplicates"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(args.splits_file).read_text())
    pairs = find_near_duplicates(splits) if args.exclude is None else [(c, []) for c in args.exclude]
    exclude = {c for c, _ in pairs}
    print("Test classes with a training-class sibling:", pairs or "none")
    if not exclude:
        return

    state = json.loads((root / "state.json").read_text())
    tta = state["chosen"]["baseline"]["tta"]
    runs = {k: [Path(str(d).replace("\\", "/")) for d in v] for k, v in state["final_runs"].items()}
    results = {}
    for s in range(min(args.n_seeds, len(runs["baseline"]))):
        out = out_dir / f"s{s}_episodes.json"
        if out.exists():
            per_model = json.loads(out.read_text())
        else:
            per_model = {}
            for method in ("baseline", "prototype"):
                for k in (1, 5):
                    per_model[f"k{k}_{method}"] = [
                        episode_scores(runs[method][s] / "best.pt", method, splits["test"], args, k, e,
                                       tta if method == "baseline" else (0, 0))
                        for e in range(args.eval_seeds)]
            out.write_text(json.dumps(per_model))
        for key, by_eval_seed in per_model.items():
            for m in ("mIoU", "F1"):
                # Same aggregation as the main results: mean per evaluation seed, then mean over seeds.
                all_mean = statistics.mean(statistics.mean(r[m] for r in eps) for eps in by_eval_seed)
                kept = [[r[m] for r in eps if r["class"] not in exclude] for eps in by_eval_seed]
                kept_mean = statistics.mean(statistics.mean(v) for v in kept if v)
                n_removed = sum(len(eps) - len(v) for eps, v in zip(by_eval_seed, kept))
                results.setdefault((key, m), []).append((all_mean, kept_mean, n_removed))
        print(f"training seed {s} done", flush=True)

    summary = {"excluded_test_classes": pairs, "n_training_seeds": None, "results": []}
    print(f"\n{'condition':<14}{'metric':<6}{'all':>8}{'without':>9}{'change':>9}{'episodes removed':>18}")
    for (key, m), vals in sorted(results.items()):
        a = statistics.mean(v[0] for v in vals)
        b = statistics.mean(v[1] for v in vals)
        removed = vals[0][2]
        summary["n_training_seeds"] = len(vals)
        summary["results"].append({"condition": key, "metric": m, "all_classes": a, "without": b,
                                   "change": b - a, "episodes_removed_per_model": removed,
                                   "episodes_per_model": args.eval_seeds * args.episodes})
        print(f"{key:<14}{m:<6}{a:>8.4f}{b:>9.4f}{b - a:>+9.4f}{removed:>18}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
