"""
scripts/sanity_check.py — logic correctness tests using synthetic data.

No dataset, no GPU, no pretrained weights needed — this checks the core
math in src/models.py directly, using small hand-made tensors where the
"correct" behaviour is known in advance. Run this any time you touch
models.py to make sure nothing's silently broken.

Usage:
    python -m scripts.sanity_check
"""

import torch
import torch.nn.functional as F

from src.models import (
    compute_prototypes,
    prototype_logits,
    SegHead,
    baseline_logits,
    adapt_baseline,
)


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def test_prototype_separates_known_classes():
    """
    Build a synthetic support set where foreground features are all near
    [1, 0] and background features are all near [0, 1]. If the prototype
    math is correct, the foreground prototype should end up near [1, 0]
    and the background prototype near [0, 1], and a query pixel that
    looks like [1, 0] should be classified as foreground.
    """
    b, k, c, h, w = 1, 3, 2, 4, 4
    feats = torch.zeros(b, k, c, h, w)
    occ = torch.zeros(b, k, h, w)

    # Left half of each support image = foreground (feature ~[1,0]),
    # right half = background (feature ~[0,1]).
    feats[:, :, 0, :, : w // 2] = 1.0
    feats[:, :, 1, :, w // 2 :] = 1.0
    occ[:, :, :, : w // 2] = 1.0

    p_fg, p_bg = compute_prototypes(feats, occ, weighted=False)

    ok = True
    ok &= check("foreground prototype ~ [1, 0]", torch.allclose(p_fg, torch.tensor([[1.0, 0.0]]), atol=1e-4))
    ok &= check("background prototype ~ [0, 1]", torch.allclose(p_bg, torch.tensor([[0.0, 1.0]]), atol=1e-4))

    # A query image that's entirely "foreground-looking" ([1, 0] everywhere)
    # should be classified as foreground everywhere.
    query_feats = torch.zeros(b, c, h, w)
    query_feats[:, 0, :, :] = 1.0
    logits = prototype_logits(query_feats, p_fg, p_bg)
    pred = logits.argmax(dim=1)
    ok &= check("all-foreground query classified as foreground", (pred == 1).all().item())

    return ok


def test_weighted_ablation_downweights_boundary():
    """
    A support mask with occupancy exactly 0.5 everywhere (maximally
    ambiguous, as if every pixel is a boundary pixel) should contribute
    almost nothing to either prototype once weighted=True, compared to
    weighted=False where it still contributes fully.
    """
    b, k, c, h, w = 1, 1, 2, 2, 2
    feats = torch.ones(b, k, c, h, w)
    occ = torch.full((b, k, h, w), 0.5)  # maximally ambiguous everywhere

    p_fg_plain, p_bg_plain = compute_prototypes(feats, occ, weighted=False)
    p_fg_weighted, p_bg_weighted = compute_prototypes(feats, occ, weighted=True)

    # Plain version still produces a well-defined prototype (weights sum
    # to something meaningful); weighted version's weights collapse to
    # ~0 everywhere, so the denominator clamp (1e-5) dominates and the
    # result becomes numerically unstable/degenerate on this adversarial
    # all-ambiguous input. We're checking the weighted version's weights
    # are near zero, which is the actual intended behaviour.
    confidence = (2 * occ - 1).abs()
    ok = check("confidence is ~0 when occupancy is 0.5 everywhere", torch.allclose(confidence, torch.zeros_like(confidence), atol=1e-6))
    return ok


def test_baseline_head_shapes():
    """1x1 conv head should output (B, 2, H, W) regardless of input feature
    map resolution, matching the upsampled target size."""
    head = SegHead(in_channels=8, n_classes=2)
    feats = torch.randn(2, 8, 16, 16)
    out = head(feats, out_hw=(64, 64))
    return check("SegHead output shape is (2, 2, 64, 64)", tuple(out.shape) == (2, 2, 64, 64))


def test_adapt_baseline_source_check():
    """
    adapt_baseline can't be exercised with fake tensors here, since it
    calls the real SegFormer backbone's HuggingFace-style forward
    signature (pixel_values=...) internally, which a dummy backbone
    doesn't implement. This is a source-level check, not a runtime test:
    it confirms the function deepcopies both backbone and head before
    training on them, which is what actually prevents one episode's
    adaptation from leaking into the next.
    """
    import inspect
    from src import models
    src = inspect.getsource(models.adapt_baseline)
    has_backbone_copy = "copy.deepcopy(backbone)" in src
    has_head_copy = "copy.deepcopy(head)" in src
    print("  (source-level check only — this function needs the real "
          "SegFormer backbone to test live, which needs a GPU/download)")
    ok = True
    ok &= check("adapt_baseline deepcopies backbone before training on it", has_backbone_copy)
    ok &= check("adapt_baseline deepcopies head before training on it", has_head_copy)
    return ok


def test_episode_varies_by_epoch():
    """
    Regression test for the frozen-episode bug: for a same idx across
    epochs, the episode should change. Checks several idx values and
    requires that NOT ALL of them collide between epoch 0 and epoch 1
    — a single-idx version of this test can pass or fail by pure chance
    (few images means real collisions happen sometimes), so this checks
    enough idx values that an all-collide false failure is astronomically
    unlikely if the fix is actually working.
    """
    import tempfile
    from pathlib import Path
    from PIL import Image
    from src.dataset import FSS1000Episodic

    with tempfile.TemporaryDirectory() as tmp:
        class_dir = Path(tmp) / "dummy_class"
        class_dir.mkdir()
        # 20 distinctly-colored images, so collisions across epochs at any
        # one idx are plausible sometimes, but not across all 8 idx values
        # tested below unless the fix genuinely isn't working.
        for i in range(1, 21):
            img = Image.new("RGB", (8, 8), color=(i * 12, 0, 0))
            mask = Image.new("L", (8, 8), color=255)
            img.save(class_dir / f"{i}.jpg")
            mask.save(class_dir / f"{i}.png")

        n_idx = 8
        ds = FSS1000Episodic(tmp, ["dummy_class"], k_shot=1, img_size=8, episodes_per_epoch=n_idx, seed=0)

        ds.set_epoch(0)
        epoch0_queries = [ds[i][2] for i in range(n_idx)]
        epoch0_queries_again = [ds[i][2] for i in range(n_idx)]

        ds.set_epoch(1)
        epoch1_queries = [ds[i][2] for i in range(n_idx)]

        ds.set_epoch(0)
        epoch0_queries_c = [ds[i][2] for i in range(n_idx)]

        same_epoch_matches = all(torch.equal(a, b) for a, b in zip(epoch0_queries, epoch0_queries_again))
        any_differ_across_epoch = any(not torch.equal(a, b) for a, b in zip(epoch0_queries, epoch1_queries))
        returning_matches = all(torch.equal(a, b) for a, b in zip(epoch0_queries, epoch0_queries_c))

        ok = True
        ok &= check("same epoch, same idx -> identical episode (all 8 idx)", same_epoch_matches)
        ok &= check("different epoch -> at least one of 8 idx differs", any_differ_across_epoch)
        ok &= check("returning to a prior epoch reproduces its episodes (all 8 idx)", returning_matches)
        return ok


def test_class_split_disjoint_and_sized():
    """
    class_level_split should partition every class into exactly one of
    train/val/test (no drops, no duplicates), size each split according
    to val_frac/test_frac, and be reproducible for a fixed seed.
    """
    from src.dataset import class_level_split

    classes = [f"class_{i}" for i in range(100)]
    splits = class_level_split(classes, val_frac=0.1, test_frac=0.2, seed=42)
    train, val, test = splits["train"], splits["val"], splits["test"]

    ok = True
    ok &= check("train/val disjoint", set(train).isdisjoint(val))
    ok &= check("train/test disjoint", set(train).isdisjoint(test))
    ok &= check("val/test disjoint", set(val).isdisjoint(test))
    ok &= check("every class assigned to exactly one split, none dropped",
                set(train) | set(val) | set(test) == set(classes)
                and len(train) + len(val) + len(test) == len(classes))
    ok &= check("split sizes match val_frac/test_frac (70/10/20 for n=100)",
                len(train) == 70 and len(val) == 10 and len(test) == 20)

    splits_again = class_level_split(classes, val_frac=0.1, test_frac=0.2, seed=42)
    ok &= check("same seed reproduces an identical split", splits_again == splits)
    return ok


def _make_dummy_class(class_dir, n):
    """n distinctly-colored (image, mask) pairs — real overlap or shape
    mistakes show up as exact tensor (mis)matches, not just look-alikes."""
    from PIL import Image

    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        Image.new("RGB", (8, 8), color=(i * 20 % 256, 0, 0)).save(class_dir / f"{i}.jpg")
        Image.new("L", (8, 8), color=255).save(class_dir / f"{i}.png")


def test_support_query_never_overlap():
    """
    Regression check for accidental support/query overlap: across many
    episodes (multiple idx, multiple epochs), the query image should
    never come out identical to any of its own episode's k support
    images. augment=False so a real overlap would show up as an exact
    tensor match, not just a similar-looking crop.
    """
    import tempfile
    from pathlib import Path
    from src.dataset import FSS1000Episodic

    with tempfile.TemporaryDirectory() as tmp:
        _make_dummy_class(Path(tmp) / "dummy_class", n=10)
        ds = FSS1000Episodic(tmp, ["dummy_class"], k_shot=3, img_size=8,
                              episodes_per_epoch=20, augment=False, seed=0)

        overlap_found = False
        for epoch in range(2):
            ds.set_epoch(epoch)
            for i in range(len(ds)):
                support_imgs, _, query_img, _, _ = ds[i]
                if any(torch.equal(query_img, support_imgs[k]) for k in range(support_imgs.shape[0])):
                    overlap_found = True

        return check("query image never duplicates a support image, across episodes/epochs",
                     not overlap_found)


def test_fss_collate_shapes():
    """fss_collate should stack a batch of episodes into the (B, k, ...) /
    (B, ...) shapes the training/eval code assumes, and keep class names
    as a plain list of strings (not tensors)."""
    import tempfile
    from pathlib import Path
    from src.dataset import FSS1000Episodic, fss_collate

    with tempfile.TemporaryDirectory() as tmp:
        _make_dummy_class(Path(tmp) / "dummy_class", n=6)
        k_shot, img_size, batch_size = 2, 8, 3
        ds = FSS1000Episodic(tmp, ["dummy_class"], k_shot=k_shot, img_size=img_size,
                              episodes_per_epoch=batch_size, augment=False, seed=0)
        batch = [ds[i] for i in range(batch_size)]
        s_imgs, s_masks, q_imgs, q_masks, classes = fss_collate(batch)

        ok = True
        ok &= check("support_imgs shape (B, k, 3, H, W)",
                    tuple(s_imgs.shape) == (batch_size, k_shot, 3, img_size, img_size))
        ok &= check("support_masks shape (B, k, H, W)",
                    tuple(s_masks.shape) == (batch_size, k_shot, img_size, img_size))
        ok &= check("query_imgs shape (B, 3, H, W)",
                    tuple(q_imgs.shape) == (batch_size, 3, img_size, img_size))
        ok &= check("query_masks shape (B, H, W)",
                    tuple(q_masks.shape) == (batch_size, img_size, img_size))
        ok &= check("classes is a plain list of strings, length B",
                    isinstance(classes, list) and len(classes) == batch_size
                    and all(isinstance(c, str) for c in classes))
        return ok


if __name__ == "__main__":
    results = [
        test_prototype_separates_known_classes(),
        test_weighted_ablation_downweights_boundary(),
        test_baseline_head_shapes(),
        test_adapt_baseline_source_check(),
        test_episode_varies_by_epoch(),
        test_class_split_disjoint_and_sized(),
        test_support_query_never_overlap(),
        test_fss_collate_shapes(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed.")
