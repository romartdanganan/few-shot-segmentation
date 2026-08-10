"""
src/dataset.py — FSS-1000 episodic dataset loader.

Expected folder layout (this is FSS-1000's standard release layout):

    <data_root>/
        <class_name_1>/
            1.jpg
            1.png   <- binary mask, same stem as its image
            2.jpg
            2.png
            ...
        <class_name_2>/
            ...

If your download differs (e.g. images/masks split into separate
subfolders), adjust `_list_pairs()` below — everything else in this
file is independent of that detail.

Class-level split: classes are split into train/val/test *once*, with a
fixed seed, so a class never appears in more than one role (this is the
leakage safeguard described in the design report, Section III-A).
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGE_EXTS = {".jpg", ".jpeg"}
MASK_EXT = ".png"


def _list_pairs(class_dir: Path):
    """Return sorted list of (image_path, mask_path) for one class folder."""
    pairs = []
    for img_path in sorted(class_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        mask_path = img_path.with_suffix(MASK_EXT)
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def discover_classes(data_root):
    """List every class folder that actually contains at least 2 valid
    image/mask pairs (need >=1 support + 1 query)."""
    data_root = Path(data_root)
    classes = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and len(_list_pairs(d)) >= 2:
            classes.append(d.name)
    return classes


def class_level_split(classes, val_frac=0.1, test_frac=0.2, seed=42):
    """Split class names into train/val/test with a fixed seed, so the
    split is reproducible across runs and nobody has to remember to pass
    the same random state twice."""
    rng = random.Random(seed)
    shuffled = classes[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    return {"train": train, "val": val, "test": test}


class FSS1000Episodic(Dataset):
    """Each item is one episode: k support (image, binary mask) pairs and
    one query (image, binary mask) pair, all for a single randomly-sampled
    class from the given split.
    """

    def __init__(self, data_root, class_names, k_shot=5, img_size=256,
                 episodes_per_epoch=200, augment=False, seed=0):
        self.data_root = Path(data_root)
        self.class_names = class_names
        self.k_shot = k_shot
        self.img_size = img_size
        self.episodes_per_epoch = episodes_per_epoch
        self.augment = augment
        self.seed = seed

        self.class_pairs = {
            c: _list_pairs(self.data_root / c) for c in class_names
        }
        too_small = [c for c, p in self.class_pairs.items() if len(p) < k_shot + 1]
        if too_small:
            raise ValueError(
                f"{len(too_small)} class(es) have fewer than k_shot+1 images, "
                f"e.g. {too_small[:3]}. Reduce --k-shot or drop these classes."
            )

    def __len__(self):
        return self.episodes_per_epoch

    def _load(self, img_path, mask_path, gen):
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            if torch.rand(1, generator=gen).item() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # mild scale jitter: random-resized-crop back to img_size
            scale = 0.85 + 0.3 * torch.rand(1, generator=gen).item()
            new_size = max(8, int(self.img_size * scale))
            img = TF.resize(img, [new_size, new_size])
            mask = TF.resize(mask, [new_size, new_size], interpolation=TF.InterpolationMode.NEAREST)
            img = TF.center_crop(img, [self.img_size, self.img_size])
            mask = TF.center_crop(mask, [self.img_size, self.img_size])
            if torch.rand(1, generator=gen).item() < 0.5:
                jitter = 0.8 + 0.4 * torch.rand(1, generator=gen).item()
                img = TF.adjust_brightness(img, jitter)

        img_t = TF.to_tensor(img)

        # Match the preprocessing used by the ADE20K-pretrained SegFormer checkpoint.
        img_t = TF.normalize(
            img_t,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_t = (TF.to_tensor(mask) > 0.5).float().squeeze(0)
        return img_t, mask_t

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(self.seed * 100000 + idx)
        cls = self.class_names[torch.randint(len(self.class_names), (1,), generator=gen).item()]
        pairs = self.class_pairs[cls]

        order = torch.randperm(len(pairs), generator=gen).tolist()
        support_idx = order[:self.k_shot]
        query_idx = order[self.k_shot]

        support_imgs, support_masks = [], []
        for i in support_idx:
            im, mk = self._load(*pairs[i], gen)
            support_imgs.append(im)
            support_masks.append(mk)
        query_img, query_mask = self._load(*pairs[query_idx], gen)

        return (
            torch.stack(support_imgs), torch.stack(support_masks),
            query_img, query_mask, cls,
        )


def fss_collate(batch):
    """Default collate but keeps the class-name strings as a plain list."""
    support_imgs = torch.stack([b[0] for b in batch])
    support_masks = torch.stack([b[1] for b in batch])
    query_imgs = torch.stack([b[2] for b in batch])
    query_masks = torch.stack([b[3] for b in batch])
    classes = [b[4] for b in batch]
    return support_imgs, support_masks, query_imgs, query_masks, classes
