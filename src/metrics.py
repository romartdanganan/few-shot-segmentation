"""
src/metrics.py — mIoU and F1 for binary (foreground vs. background)
segmentation, matching the metrics named in the Design Report (Section
III-D / IV).
"""

import torch


@torch.no_grad()
def binary_mask_metrics(logits, target_mask, eps=1e-7):
    """logits: (B, 2, H, W); target_mask: (B, H, W) in {0, 1}.
    Returns dict of per-batch-mean mIoU and F1 (foreground class).
    """
    pred = logits.argmax(dim=1)
    target = target_mask.long()

    tp = ((pred == 1) & (target == 1)).sum(dim=(1, 2)).float()
    fp = ((pred == 1) & (target == 0)).sum(dim=(1, 2)).float()
    fn = ((pred == 0) & (target == 1)).sum(dim=(1, 2)).float()
    tn = ((pred == 0) & (target == 0)).sum(dim=(1, 2)).float()

    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = (iou_fg + iou_bg) / 2

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "mIoU": miou.mean().item(),
        "F1": f1.mean().item(),
    }


class RunningStats:
    """Accumulates per-episode metric values so you can report mean +/-
    standard deviation across episodes/seeds, as required by the
    evaluation protocol in the design report."""

    def __init__(self):
        self.values = {}

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self.values.setdefault(k, []).append(v)

    def summary(self):
        import statistics
        out = {}
        for k, vals in self.values.items():
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out[k] = (mean, std)
        return out
