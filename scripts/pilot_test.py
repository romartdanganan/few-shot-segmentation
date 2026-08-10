"""
pilot_test.py — AIML339 feasibility pilot

Purpose (matches Design Report, Section III-E "Feasibility and Pilot Plan",
and Aaron's approval-email advice): before committing to the full training
schedule, confirm (1) the training pipeline is mechanically correct for both
methods, and (2) it fits in 8 GB of VRAM with an acceptable per-epoch time on
a GTX 3070.

It implements both conditions from Table I of the report on a *synthetic*
few-shot dataset (random background + a random blob "object"), so you can
run this today, before FSS-1000 is wired up, and get a real timing/VRAM
number back in minutes:

  Method A (baseline):    SegFormer MiT-B0 + 1x1 conv head, fine-tuned with
                           cross-entropy on the k-shot support set.
  Method B (proposed):    Episodic training. Class prototype via masked
                           average pooling (Eq. 1), query pixels classified
                           by negative squared distance to the prototype
                           (Eq. 2). Optional distance-weighted ablation.

Once this confirms the pipeline works and fits in memory, swap
`SyntheticFewShotDataset` for a real FSS-1000 loader (see the TODO near the
bottom) without touching the model or training code.

Usage:
    python pilot_test.py --method baseline   --img-size 256 --episodes 20
    python pilot_test.py --method prototype  --img-size 256 --episodes 20
    python pilot_test.py --method prototype  --img-size 512 --episodes 20 --weighted
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Synthetic episodic dataset — stand-in for FSS-1000 while I validate the
# pipeline. Each "episode" is one class: k support (image, binary mask)
# pairs plus one query (image, binary mask) pair. The object is a random
# blob so the model has something non-trivial, but consistent, to segment.
# --------------------------------------------------------------------------
class SyntheticFewShotDataset(torch.utils.data.Dataset):
    def __init__(self, n_episodes, k_shot, img_size):
        self.n_episodes = n_episodes
        self.k_shot = k_shot
        self.img_size = img_size

    def __len__(self):
        return self.n_episodes

    def _make_pair(self, gen):
        s = self.img_size
        # background: smooth random noise
        img = torch.rand(3, s, s, generator=gen) * 0.4 + 0.3
        # foreground: a random-radius, random-position blob
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij"
        )
        cx, cy = (torch.rand(2, generator=gen) * 1.2 - 0.6).tolist()
        r = 0.25 + 0.25 * torch.rand(1, generator=gen).item()
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).float()
        # give the object a distinct colour so there's real signal to learn
        colour = torch.rand(3, 1, 1, generator=gen) * 0.5 + 0.5
        img = img * (1 - mask) + colour * mask
        return img, mask

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(idx)
        support_imgs, support_masks = [], []
        for _ in range(self.k_shot):
            im, mk = self._make_pair(gen)
            support_imgs.append(im)
            support_masks.append(mk)
        query_img, query_mask = self._make_pair(gen)
        return (
            torch.stack(support_imgs),          # (k, 3, H, W)
            torch.stack(support_masks),          # (k, H, W)
            query_img,                           # (3, H, W)
            query_mask,                          # (H, W)
        )


# --------------------------------------------------------------------------
# Backbone — matches Design Report III-A: pretrained SegFormer MiT-B0,
# shared between both methods.
# --------------------------------------------------------------------------
def build_backbone():
    backbone = SegformerModel.from_pretrained("nvidia/mit-b0")
    return backbone.to(DEVICE)


def extract_features(backbone, images):
    """images: (B, 3, H, W) in [0, 1]. Returns (B, C, h, w) feature map."""
    out = backbone(pixel_values=images).last_hidden_state
    return out


# --------------------------------------------------------------------------
# Method A: baseline — cross-entropy fine-tuning with a small conv head.
# --------------------------------------------------------------------------
class SegHead(nn.Module):
    def __init__(self, in_channels, n_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, feats, out_hw):
        logits = self.conv(feats)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


def baseline_step(backbone, head, support_imgs, support_masks):
    """Fine-tune directly on the k-shot support set (Table I, column A)."""
    b, k, c, h, w = support_imgs.shape
    imgs = support_imgs.view(b * k, c, h, w)
    masks = support_masks.view(b * k, h, w).long()

    feats = extract_features(backbone, imgs)
    logits = head(feats, (h, w))
    loss = F.cross_entropy(logits, masks)
    return loss


# --------------------------------------------------------------------------
# Method B: proposed — prototype-based episodic training (Eq. 1 and 2).
# --------------------------------------------------------------------------
def compute_prototypes(support_feats, support_occupancy, weighted=False):
    """Eq. (1): p_c = (1/|S_c|) * sum f_theta(x) . y, masked average pooling.
    support_feats: (B, k, C, h, w).
    support_occupancy: (B, k, h, w) in [0, 1] — the *fraction* of each
        low-res cell covered by the mask (from area/bilinear downsampling,
        not nearest-neighbour), which is ~0 or ~1 away from the object
        boundary and close to 0.5 right on it.
    weighted: if True, applies the distance-weighted ablation from the
        report — support pixels near the mask boundary (occupancy close to
        0.5, i.e. low confidence) contribute less to the prototype.
    Returns fg/bg prototypes of shape (B, C).
    """
    b, k, c, h, w = support_feats.shape
    feats = support_feats.view(b, k * h * w, c)
    occ = support_occupancy.view(b, k * h * w, 1)
    fg_weight, bg_weight = occ, 1 - occ

    if weighted:
        confidence = (2 * occ - 1).abs()  # 1 = confidently fg/bg, 0 = right on the boundary
        fg_weight = fg_weight * confidence
        bg_weight = bg_weight * confidence

    def pool(weight):
        num = (feats * weight).sum(dim=1)
        den = weight.sum(dim=1).clamp(min=1e-5)
        return num / den

    return pool(fg_weight), pool(bg_weight)


def prototype_logits(query_feats, p_fg, p_bg):
    """Eq. (2): d(q, c) = -||f_theta(q) - p_c||^2, used directly as logits.
    query_feats: (B, C, h, w); p_fg/p_bg: (B, C).
    Distances are scaled by 1/sqrt(C) — a standard prototypical-network
    stabiliser (Snell et al., 2017) that keeps early-training logits from
    saturating the softmax when C is large, without changing what the
    method computes.
    """
    c = query_feats.shape[1]
    scale = c ** 0.5

    def neg_sq_dist(feats, proto):
        proto = proto.view(proto.shape[0], proto.shape[1], 1, 1)
        return -((feats - proto) ** 2).sum(dim=1, keepdim=True) / scale

    d_fg = neg_sq_dist(query_feats, p_fg)
    d_bg = neg_sq_dist(query_feats, p_bg)
    return torch.cat([d_bg, d_fg], dim=1)  # (B, 2, h, w)


def prototype_step(backbone, support_imgs, support_masks, query_img, query_mask, weighted=False):
    """Episodic step (Table I, column B)."""
    b, k, c, h, w = support_imgs.shape
    flat_imgs = support_imgs.view(b * k, c, h, w)
    s_feats = extract_features(backbone, flat_imgs)
    fc, fh, fw = s_feats.shape[1], s_feats.shape[2], s_feats.shape[3]
    s_feats = s_feats.view(b, k, fc, fh, fw)

    # "area" downsampling gives a fractional occupancy per low-res cell
    # (not just 0/1), which the weighted ablation uses as its confidence
    # signal near mask boundaries.
    s_occ = F.interpolate(support_masks.view(b * k, 1, h, w), size=(fh, fw), mode="area")
    s_occ = s_occ.view(b, k, fh, fw)

    p_fg, p_bg = compute_prototypes(s_feats, s_occ, weighted=weighted)

    q_feats = extract_features(backbone, query_img)  # (b, fc, fh, fw)
    logits = prototype_logits(q_feats, p_fg, p_bg)
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

    loss = F.cross_entropy(logits, query_mask.long())
    return loss


# --------------------------------------------------------------------------
# Pilot loop: run a handful of episodes, time them, and report peak VRAM.
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "prototype"], required=True)
    ap.add_argument("--img-size", type=int, default=256, help="e.g. 256 or 512")
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=20, help="how many episodes to time")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--weighted", action="store_true", help="prototype method: use distance-weighted ablation")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print("Loading pretrained SegFormer MiT-B0 backbone...")
    backbone = build_backbone()
    backbone.train()

    params = list(backbone.parameters())
    head = None
    if args.method == "baseline":
        # infer channel count from the backbone config
        c_out = backbone.config.hidden_sizes[-1]
        head = SegHead(c_out).to(DEVICE)
        params += list(head.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and DEVICE.type == "cuda"))

    # TODO: once this pilot confirms feasibility, replace this dataset with
    # a real FSS-1000 loader: point it at a downloaded copy of the dataset
    # (https://github.com/HKUSTCV/FSS-1000, mirrored on Kaggle), sample a
    # class-folder per episode, and use the official image/mask filenames
    # as support/query according to your train/val/test class-level split.
    dataset = SyntheticFewShotDataset(args.episodes, args.k_shot, args.img_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    losses = []
    times = []
    print(f"\nRunning {args.episodes} episodes | method={args.method} | "
          f"img_size={args.img_size} | k_shot={args.k_shot} | amp={args.amp}\n")

    for step, (s_imgs, s_masks, q_img, q_mask) in enumerate(loader):
        s_imgs, s_masks = s_imgs.to(DEVICE), s_masks.to(DEVICE)
        q_img, q_mask = q_img.to(DEVICE), q_mask.to(DEVICE)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        opt.zero_grad()
        with torch.autocast(device_type=DEVICE.type, enabled=(args.amp and DEVICE.type == "cuda")):
            if args.method == "baseline":
                loss = baseline_step(backbone, head, s_imgs, s_masks)
            else:
                loss = prototype_step(backbone, s_imgs, s_masks, q_img, q_mask, weighted=args.weighted)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        losses.append(loss.item())
        times.append(dt)
        print(f"  episode {step+1:>3}/{args.episodes}  loss={loss.item():.4f}  time={dt:.3f}s")

    print("\n----- pilot summary -----")
    print(f"mean loss, first 5 episodes:  {sum(losses[:5]) / min(5, len(losses)):.4f}")
    print(f"mean loss, last 5 episodes:   {sum(losses[-5:]) / min(5, len(losses)):.4f}")
    if losses[-1] < losses[0]:
        print("Loss decreased over the pilot run — the training signal is real, not noise.")
    else:
        print("Loss did NOT decrease — before scaling up, check the learning rate, "
              "the mask/label indexing, and that gradients are actually flowing.")

    mean_time = sum(times[1:]) / max(1, len(times) - 1)  # skip first (warm-up) step
    print(f"\nmean time/episode (after warm-up): {mean_time:.3f}s")
    print(f"projected time for 1000 episodes:  {mean_time * 1000 / 60:.1f} min")

    if DEVICE.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"\npeak VRAM used: {peak_mb:.0f} MB  /  {total_mb:.0f} MB available")
        headroom = total_mb - peak_mb
        print(f"headroom: {headroom:.0f} MB")
        if headroom < 1000:
            print("Less than ~1 GB headroom — drop --img-size, or add gradient "
                  "accumulation, before running the full schedule.")
        else:
            print("Comfortable headroom at this image size and batch size.")
    else:
        print("\nNo CUDA device found — this run only validated the pipeline logic, "
              "not GPU timing/VRAM. Re-run on the GTX 3070 for real numbers.")


if __name__ == "__main__":
    main()
