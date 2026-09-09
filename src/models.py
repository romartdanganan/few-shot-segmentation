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
import copy
from transformers import SegformerForSemanticSegmentation


# Pretrained SegFormer-B0 checkpoint, already fine-tuned on ADE20K (a general
# scene-segmentation dataset). We reuse it as a starting point rather than
# training a backbone from scratch, since the whole point of this project is
# to compare *training objectives* on top of a shared, capable backbone.
ADE20K_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"


def build_backbone(pretrained_name=ADE20K_CHECKPOINT):
    """
    Load SegFormer-B0 fine-tuned on ADE20K and return its MiT encoder.

    Both the baseline and prototype methods use this exact same encoder
    initialization, matching the design report.
    """
    # HuggingFace's SegformerForSemanticSegmentation bundles an encoder
    # (the actual MiT transformer backbone) with a decoder head trained for
    # ADE20K's specific classes. We only want the encoder part, since both
    # of our methods build their own classification logic on top of it.
    full_model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name
    )

    return full_model.segformer


def extract_features(backbone, images):
    """images: (B, 3, H, W) in [0, 1]. Returns (B, C, h, w) feature map."""
    # This is the "turn pixels into feature vectors" step described in the
    # design report. The output has a smaller height/width than the input
    # (SegFormer downsamples internally) but many more channels (C) per
    # spatial location — each of those C-length vectors is one pixel's
    # "feature vector", i.e. its position in the model's learned semantic
    # space. Two locations with similar vectors are things the model
    # considers semantically similar.
    return backbone(pixel_values=images).last_hidden_state


class SegHead(nn.Module):
    """1x1 conv head for the baseline (Table I, column A)."""

    def __init__(self, in_channels, n_classes=2):
        super().__init__()
        # A 1x1 convolution here is really just a small trainable linear
        # classifier applied independently at every pixel location: it
        # takes each pixel's feature vector (in_channels numbers) and maps
        # it to n_classes scores (foreground vs. background). This is the
        # only part of the baseline that has its own learned weights beyond
        # the shared backbone.
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, feats, out_hw):
        logits = self.conv(feats)
        # The backbone's feature map is smaller than the original image, so
        # we upsample the per-pixel class scores back up to the original
        # image resolution before comparing against the ground-truth mask.
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


def baseline_logits(backbone, head, images):
    """
    Run images through the shared encoder and baseline segmentation head.

    images: (B, 3, H, W)
    returns: (B, 2, H, W)
    """
    h, w = images.shape[-2:]
    feats = extract_features(backbone, images)
    return head(feats, (h, w))


def baseline_loss(backbone, head, support_imgs, support_masks):
    """
    Fine-tune the baseline directly on the k-shot support set
    using cross-entropy, as described in the design report.
    """
    # support_imgs comes in as (batch, k_shot, channels, height, width) —
    # one batch of episodes, each with k support images. We flatten the
    # batch and k-shot dimensions together so every support image is just
    # treated as one more ordinary training image; the baseline doesn't
    # care that these images came from a "few-shot episode" at all, it just
    # does standard supervised learning on whatever images it's given.
    b, k, c, h, w = support_imgs.shape

    imgs = support_imgs.view(b * k, c, h, w)
    masks = support_masks.view(b * k, h, w).long()

    logits = baseline_logits(backbone, head, imgs)

    # Cross-entropy compares the predicted per-pixel class scores against
    # the real ground-truth mask. This is the actual training signal: the
    # gradient of this loss flows back through the head and the backbone,
    # updating both so the prediction gets closer to the true mask next time.
    return F.cross_entropy(logits, masks)


def baseline_query_logits(backbone, head, query_img):
    """
    Predict the segmentation mask for the held-out query image
    after the baseline has been adapted on the support set.
    """
    return baseline_logits(backbone, head, query_img)

def adapt_baseline(backbone,head,support_imgs,support_masks,lr=1e-4,steps=5):
    """
    Create an episode-specific copy of the baseline and fine-tune it on
    the k-shot support set.

    The original backbone/head are untouched so every novel episode
    starts from the same checkpoint.
    """
    # Unlike the prototype method, the baseline has no built-in way to
    # handle a brand-new class at evaluation time — its classifier head was
    # trained on the training classes only. So at eval time we take a
    # *copy* of the trained model and give it a handful of extra gradient
    # steps on this specific episode's support set, simulating "quickly
    # fine-tuning on the few examples you have for this new class."
    # We deepcopy first so this adaptation doesn't permanently change the
    # original model — the next episode (a different novel class) needs to
    # start from the same clean checkpoint, not from wherever the last
    # episode's adaptation left off.
    adapted_backbone = copy.deepcopy(backbone)
    adapted_head = copy.deepcopy(head)

    adapted_backbone.train()
    adapted_head.train()

    optimizer = torch.optim.AdamW(
        list(adapted_backbone.parameters()) + list(adapted_head.parameters()),
        lr=lr,
    )

    # A handful of ordinary gradient-descent steps: compute the loss on the
    # support set, backpropagate, update weights, repeat.
    for _ in range(steps):
        optimizer.zero_grad()

        loss = baseline_loss(
            adapted_backbone,
            adapted_head,
            support_imgs,
            support_masks,
        )

        loss.backward()
        optimizer.step()

    # Switch back to eval mode (disables things like dropout) before this
    # adapted copy gets used to actually predict on the query image.
    adapted_backbone.eval()
    adapted_head.eval()

    return adapted_backbone, adapted_head

def compute_prototypes(support_feats, support_occupancy, weighted=False):
    """Eq. (1): masked average pooling. See pilot_test.py for full derivation
    of the distance-weighted ablation variant."""
    # support_feats: (batch, k_shot, channels, feat_h, feat_w) — the feature
    # vector at every spatial location of every support image.
    # support_occupancy: same spatial shape, but 1 channel — how much of
    # each feature-map location is covered by the foreground mask, as a
    # fraction between 0 and 1 (it's fractional, not strictly 0/1, because
    # the original full-resolution mask gets downsampled to match the
    # smaller feature map — see prototype_loss below for that step).
    b, k, c, h, w = support_feats.shape

    # Flatten k-shot and spatial dimensions together: we don't care which
    # image or which exact location a feature vector came from, only
    # whether it's a foreground vector or a background vector, since we're
    # about to average all foreground vectors into one prototype and all
    # background vectors into another.
    feats = support_feats.view(b, k * h * w, c)
    occ = support_occupancy.view(b, k * h * w, 1)

    # occ is already "how foreground is this location" (close to 1 = very
    # foreground, close to 0 = very background), so it doubles directly as
    # the averaging weight for the foreground prototype. 1 - occ is the
    # matching weight for the background prototype.
    fg_weight, bg_weight = occ, 1 - occ

    if weighted:
        # The distance-weighted ablation: locations near the mask boundary
        # have occ close to 0.5 (ambiguous — part object, part background),
        # while locations clearly inside or outside the object have occ
        # close to 1 or 0. confidence is high (near 1) for the unambiguous
        # locations and low (near 0) for the ambiguous boundary locations,
        # so multiplying it in down-weights exactly the noisy boundary
        # pixels without touching the confident interior/exterior ones.
        confidence = (2 * occ - 1).abs()
        fg_weight = fg_weight * confidence
        bg_weight = bg_weight * confidence

    def pool(weight):
        # Weighted average of feature vectors: sum(vector * weight) /
        # sum(weight). This is the actual "masked average pooling" from
        # the design report — the prototype is just a weighted mean vector.
        num = (feats * weight).sum(dim=1)
        den = weight.sum(dim=1).clamp(min=1e-5)  # avoid divide-by-zero
        return num / den

    return pool(fg_weight), pool(bg_weight)


def prototype_logits(query_feats, p_fg, p_bg):
    """Eq. (2): negative squared distance, scaled by 1/sqrt(C) for stability."""
    c = query_feats.shape[1]
    # Dividing by sqrt(channel count) keeps the distance values in a
    # reasonable numeric range regardless of how many feature channels the
    # backbone uses — without this, larger C makes the raw squared
    # distances much bigger, which can make training unstable.
    scale = c ** 0.5

    def neg_sq_dist(feats, proto):
        # proto is one vector per batch item; reshape it so it can be
        # subtracted from every spatial location of the query feature map
        # via broadcasting.
        proto = proto.view(proto.shape[0], proto.shape[1], 1, 1)
        # Negative squared distance: closer to the prototype -> less
        # negative -> higher score. This is what turns "distance to
        # prototype" into something that behaves like a classification
        # score (higher = more likely this class), which is what
        # cross-entropy expects.
        return -((feats - proto) ** 2).sum(dim=1, keepdim=True) / scale

    d_fg = neg_sq_dist(query_feats, p_fg)
    d_bg = neg_sq_dist(query_feats, p_bg)
    # Stack into a 2-class score map (background score, foreground score)
    # at every pixel, same shape as the baseline's logits — so the same
    # cross-entropy loss and evaluation code can be reused for both methods.
    return torch.cat([d_bg, d_fg], dim=1)


def prototype_loss(backbone, support_imgs, support_masks, query_img, query_mask, weighted=False):
    """Episodic loss (Table I, column B)."""
    # This function runs one full episode end-to-end: support -> prototypes
    # -> query classification -> loss. It's the training-time counterpart
    # to the walkthrough in the design report — at eval time the same
    # prototype-building and distance steps run, just without computing a
    # loss against a mask the "real world" wouldn't actually have.
    b, k, c, h, w = support_imgs.shape

    # Step 1: embed every support image with the shared backbone. Same
    # flatten-then-reshape trick as the baseline, since the backbone just
    # processes a batch of images and doesn't need to know they're grouped
    # into k-shot episodes.
    flat_imgs = support_imgs.view(b * k, c, h, w)
    s_feats = extract_features(backbone, flat_imgs)
    fc, fh, fw = s_feats.shape[1], s_feats.shape[2], s_feats.shape[3]
    s_feats = s_feats.view(b, k, fc, fh, fw)

    # Step 2: downsample the full-resolution ground-truth support masks to
    # match the (smaller) feature map resolution. mode="area" averages the
    # original mask over each downsampled cell, which is why the result is
    # a fraction (occupancy) rather than a hard 0/1 value.
    s_occ = F.interpolate(support_masks.view(b * k, 1, h, w), size=(fh, fw), mode="area")
    s_occ = s_occ.view(b, k, fh, fw)

    # Step 3: build the foreground/background prototype vectors from the
    # support features and their (given) ground-truth occupancy.
    p_fg, p_bg = compute_prototypes(s_feats, s_occ, weighted=weighted)

    # Step 4: embed the query image with the *same* backbone (same weights,
    # this is the "shared backbone" the design report keeps emphasizing —
    # support and query features live in the same space, which is what
    # makes comparing them to a prototype meaningful).
    q_feats = extract_features(backbone, query_img)

    # Step 5: classify every query pixel by distance to each prototype,
    # then upsample the resulting score map back to full image resolution.
    logits = prototype_logits(q_feats, p_fg, p_bg)
    logits_full = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

    # Step 6: compare the predicted scores to the query's real ground-truth
    # mask. This loss is what actually trains the backbone during
    # training — even though the prototype-building step itself has no
    # learned parameters of its own, the feature vectors that go into it
    # come from the backbone, so gradients flow all the way back through
    # steps 1-5 into the backbone's weights. This is how the backbone learns
    # to produce features that make this whole averaging-and-distance trick
    # work well, for classes it hasn't seen before.
    loss = F.cross_entropy(logits_full, query_mask.long())
    return loss, logits_full
