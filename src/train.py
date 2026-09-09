"""
src/train.py — full training run for either method in Table I.

This is the "real" training script that follows on from the feasibility
pilot (pilot_test.py): same backbone, same loss functions, but now
trained on real FSS-1000 data with a proper class-level train/val split,
checkpointing, and TensorBoard logging instead of a handful of synthetic
episodes.

Usage:
    python -m src.train --method baseline  --data-root data/FSS-1000 --img-size 256
    python -m src.train --method prototype --data-root data/FSS-1000 --img-size 256
    python -m src.train --method prototype --data-root data/FSS-1000 --weighted
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import FSS1000Episodic, discover_classes, class_level_split, fss_collate
from src.models import (
    build_backbone,
    SegHead,
    baseline_loss,
    baseline_query_logits,
    adapt_baseline,
    prototype_loss,
)
from src.metrics import binary_mask_metrics, RunningStats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_splits(data_root, splits_file, seed):
    splits_file = Path(splits_file)
    if splits_file.exists():
        with open(splits_file) as f:
            return json.load(f)
    classes = discover_classes(data_root)
    if len(classes) < 10:
        raise RuntimeError(
            f"Only found {len(classes)} usable class folders under {data_root}. "
            "Check --data-root points at the unzipped FSS-1000 directory."
        )
    splits = class_level_split(classes, seed=seed)
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote a fresh class-level split to {splits_file} "
          f"({len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test classes)")
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="path to the unzipped FSS-1000 folder")
    ap.add_argument("--splits-file", default="configs/class_splits.json")
    ap.add_argument("--method", choices=["baseline", "prototype"], required=True)
    ap.add_argument("--weighted", action="store_true", help="prototype method: distance-weighted ablation")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--episodes-per-epoch", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--adapt-steps",type=int,default=5,help="baseline: support-set fine-tuning steps during validation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    run_name = f"{args.method}{'_weighted' if args.weighted else ''}_k{args.k_shot}_s{args.seed}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    splits = get_splits(args.data_root, args.splits_file, args.seed)

    train_ds = FSS1000Episodic(
        args.data_root, splits["train"], k_shot=args.k_shot, img_size=args.img_size,
        episodes_per_epoch=args.episodes_per_epoch, augment=True, seed=args.seed,
    )
    val_ds = FSS1000Episodic(
        args.data_root, splits["val"], k_shot=args.k_shot, img_size=args.img_size,
        episodes_per_epoch=40, augment=False, seed=args.seed + 1000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=fss_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=fss_collate)

    print(f"Device: {DEVICE}")
    print("Loading pretrained SegFormer MiT-B0 backbone...")
    backbone = build_backbone().to(DEVICE)
    backbone.train()

    params = list(backbone.parameters())
    head = None
    if args.method == "baseline":
        c_out = backbone.config.hidden_sizes[-1]
        head = SegHead(c_out).to(DEVICE)
        params += list(head.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and DEVICE.type == "cuda"))

    best_val_miou = -1.0
    for epoch in range(args.epochs):
        backbone.train()
        if head is not None:
            head.train()
        t0 = time.time()
        epoch_loss = 0.0

        # One iteration of this loop = one batch of episodes. Each episode
        # is a (support images, support masks, one query image, one query
        # mask, class name) tuple — see FSS1000Episodic in dataset.py for
        # how these get sampled.
        for step, (s_imgs, s_masks, q_img, q_mask, cls) in enumerate(train_loader):
            s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
            q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)

            opt.zero_grad()
            # autocast runs parts of the forward pass in lower precision
            # (float16) when AMP is enabled, which uses less GPU memory and
            # is faster, at a small, generally negligible cost to accuracy.
            with torch.autocast(device_type=DEVICE.type, enabled=(args.amp and DEVICE.type == "cuda")):
                if args.method == "baseline":
                    # Baseline: cross-entropy on the support set only (see
                    # baseline_loss in models.py). The query image isn't
                    # used for training the baseline at all — the baseline
                    # is a standard supervised learner, it just happens to
                    # be trained per-episode on whatever support set it's
                    # given.
                    loss = baseline_loss(
                        backbone,
                        head,
                        s_imgs,
                        s_masks,
                    )
                else:
                    # Prototype method: one full episode as described in
                    # models.py's prototype_loss — support set builds
                    # prototypes, query gets classified against them, loss
                    # comes from comparing that classification to the
                    # query's real mask.
                    loss, _ = prototype_loss(
                        backbone,
                        s_imgs,
                        s_masks,
                        q_img,
                        q_mask,
                        weighted=args.weighted,
                    )

            # Standard PyTorch training step: backpropagate the loss, then
            # update the model's weights. scaler handles the bookkeeping
            # AMP needs to do this safely in mixed precision.
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("train/loss", loss.item(), global_step)

        mean_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs}  mean_loss={mean_loss:.4f}  time={dt:.1f}s")

        # ---- validation ----
        # Validation episodes are sampled from val-split classes, which the
        # model never trains on directly (see get_splits / class_level_split
        # in dataset.py) — this is what actually tests "few-shot on a novel
        # class", not just held-out images of already-seen classes.
        backbone.eval()
        if head is not None:
            head.eval()
        stats = RunningStats()

        for s_imgs, s_masks, q_img, q_mask, cls in val_loader:
            s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
            q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)

            if args.method == "baseline":
                # The baseline has no built-in way to adapt to a class it
                # hasn't trained on, so we give it a few gradient steps on
                # this episode's support set first (see adapt_baseline in
                # models.py) — this is the baseline's equivalent of "using
                # the k support examples", mirroring what the prototype
                # method does by building a prototype from them instead.
                adapted_backbone, adapted_head = adapt_baseline(
                    backbone,
                    head,
                    s_imgs,
                    s_masks,
                    lr=args.lr,
                    steps=args.adapt_steps,
                )

                with torch.no_grad():
                    logits = baseline_query_logits(
                        adapted_backbone,
                        adapted_head,
                        q_img,
                    )

                # Free the per-episode adapted copy — we don't want to keep
                # every episode's adapted weights in memory, and the next
                # episode should start adapting from the original trained
                # model again, not from this one's adapted state.
                del adapted_backbone
                del adapted_head

            else:
                # Prototype method needs no per-episode adaptation step —
                # building the prototype from the support set at inference
                # time already IS its way of using new examples, with no
                # gradient updates required.
                with torch.no_grad():
                    _, logits = prototype_loss(
                        backbone,
                        s_imgs,
                        s_masks,
                        q_img,
                        q_mask,
                        weighted=args.weighted,
                    )

            # binary_mask_metrics compares the predicted logits to the real
            # query mask and computes mIoU/F1 for this one episode;
            # RunningStats accumulates that across every validation episode
            # so we get an averaged score for the whole epoch.
            stats.update(binary_mask_metrics(logits, q_mask))

        summary = stats.summary()
        val_miou = summary["mIoU"][0]
        print(f"  val: mIoU={val_miou:.4f}  F1={summary['F1'][0]:.4f}")
        writer.add_scalar("val/mIoU", val_miou, epoch)
        writer.add_scalar("val/F1", summary["F1"][0], epoch)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            ckpt = {"backbone": backbone.state_dict(), "args": vars(args)}
            if head is not None:
                ckpt["head"] = head.state_dict()
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved new best checkpoint (val mIoU={val_miou:.4f})")

    print(f"\nDone. Best val mIoU: {best_val_miou:.4f}. Checkpoint + logs in {out_dir}")


if __name__ == "__main__":
    main()
