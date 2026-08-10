"""
src/models.py — shared backbone, baseline head, and prototype-based
episodic training logic (matches Design Report Table I / Eq. 1-2).

This is the same math validated in the project's feasibility pilot
(pilot_test.py) — the prototype computation, the distance-based logits,
and the 1/sqrt(C) stabiliser are unchanged from there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


def build_backbone(pretrained_name="nvidia/mit-b0"):
    """SegFormer MiT-B0 encoder, pretrained on ADE20K (matches Table I)."""
    return SegformerModel.from_pretrained(pretrained_name)


def extract_features(backbone, images):
    """images: (B, 3, H, W) in [0, 1]. Returns (B, C, h, w) feature map."""
    return backbone(pixel_values=images).last_hidden_state


class SegHead(nn.Module):
    """1x1 conv head for the baseline (Table I, column A)."""

    def __init__(self, in_channels, n_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, feats, out_hw):
        logits = self.conv(feats)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


def baseline_loss(backbone, head, support_imgs, support_masks):
    """Cross-entropy fine-tuning directly on the k-shot support set."""
    b, k, c, h, w = support_imgs.shape
    imgs = support_imgs.view(b * k, c, h, w)
    masks = support_masks.view(b * k, h, w).long()
    feats = extract_features(backbone, imgs)
    logits = head(feats, (h, w))
    return F.cross_entropy(logits, masks), logits.view(b, k, 2, h, w)[:, -1]


def compute_prototypes(support_feats, support_occupancy, weighted=False):
    """Eq. (1): masked average pooling. See pilot_test.py for full derivation
    of the distance-weighted ablation variant."""
    b, k, c, h, w = support_feats.shape
    feats = support_feats.view(b, k * h * w, c)
    occ = support_occupancy.view(b, k * h * w, 1)
    fg_weight, bg_weight = occ, 1 - occ

    if weighted:
        confidence = (2 * occ - 1).abs()
        fg_weight = fg_weight * confidence
        bg_weight = bg_weight * confidence

    def pool(weight):
        num = (feats * weight).sum(dim=1)
        den = weight.sum(dim=1).clamp(min=1e-5)
        return num / den

    return pool(fg_weight), pool(bg_weight)


def prototype_logits(query_feats, p_fg, p_bg):
    """Eq. (2): negative squared distance, scaled by 1/sqrt(C) for stability."""
    c = query_feats.shape[1]
    scale = c ** 0.5

    def neg_sq_dist(feats, proto):
        proto = proto.view(proto.shape[0], proto.shape[1], 1, 1)
        return -((feats - proto) ** 2).sum(dim=1, keepdim=True) / scale

    d_fg = neg_sq_dist(query_feats, p_fg)
    d_bg = neg_sq_dist(query_feats, p_bg)
    return torch.cat([d_bg, d_fg], dim=1)


def prototype_loss(backbone, support_imgs, support_masks, query_img, query_mask, weighted=False):
    """Episodic loss (Table I, column B)."""
    b, k, c, h, w = support_imgs.shape
    flat_imgs = support_imgs.view(b * k, c, h, w)
    s_feats = extract_features(backbone, flat_imgs)
    fc, fh, fw = s_feats.shape[1], s_feats.shape[2], s_feats.shape[3]
    s_feats = s_feats.view(b, k, fc, fh, fw)

    s_occ = F.interpolate(support_masks.view(b * k, 1, h, w), size=(fh, fw), mode="area")
    s_occ = s_occ.view(b, k, fh, fw)

    p_fg, p_bg = compute_prototypes(s_feats, s_occ, weighted=weighted)

    q_feats = extract_features(backbone, query_img)
    logits = prototype_logits(q_feats, p_fg, p_bg)
    logits_full = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

    loss = F.cross_entropy(logits_full, query_mask.long())
    return loss, logits_full
