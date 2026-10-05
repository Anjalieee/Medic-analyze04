"""
generate_datasets.py
Step 2b: Build benchmark datasets from the REAL images in data/raw.

  * COUNT sets : N images (100, 500, 1000, 5000, 10000) all at 512x512
  * SIZE sets  : 200 images at 256, 512, 1024, 2048 pixels (square)

Extra images beyond the real ones are made by light augmentation
(flip, small rotation, brightness/contrast jitter, small crop).
Only TRAIN-split images are used as sources, so the real val/test images
stay untouched for accuracy evaluation.

IMPORTANT: these generated sets are for TIMING / SCALABILITY only.
Accuracy must be reported on the real, un-augmented test split.

Usage:
    python generate_datasets.py --data data
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageEnhance

COUNTS = [100, 500, 1000, 5000, 10000]
SIZES = [256, 512, 1024, 2048]
SIZE_SET_COUNT = 200
COUNT_SET_SIZE = 512


def load_sources(data: Path):
    with open(data / "labels.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    return rows


def augment(im: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(rng.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=0)
    w, h = im.size
    c = rng.uniform(0.90, 1.0)  # small random crop
    cw, ch = int(w * c), int(h * c)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    return im


def make_set(rows, raw_dir: Path, out_dir: Path, n: int, size: int, seed: int):
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(n):
        src = rows[i % len(rows)] if i < len(rows) else rng.choice(rows)
        with Image.open(raw_dir / src["filename"]) as im:
            im = im.convert("L")
            if i >= len(rows):  # reuse of a real image -> augment it
                im = augment(im, rng)
            im = im.resize((size, size), Image.BILINEAR)
            name = f"img_{i:05d}.jpg"
            im.save(out_dir / name, quality=92)
        labels.append((name, src["label"], src["filename"]))
    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "source_image"])
        w.writerows(labels)
    print(f"  {out_dir}: {n} images @ {size}x{size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data"), type=Path)
    ap.add_argument("--skip-large", action="store_true",
                    help="skip the 10000-image and 2048px sets (disk/time)")
    args = ap.parse_args()

    rows = load_sources(args.data)
    raw_dir = args.data / "raw"
    gen = args.data / "generated"
    print(f"Source (train) images: {len(rows)}")

    print("Count sets:")
    for n in COUNTS:
        if args.skip_large and n == 10000:
            continue
        make_set(rows, raw_dir, gen / f"count_{n}", n, COUNT_SET_SIZE, seed=n)

    print("Size sets:")
    for s in SIZES:
        if args.skip_large and s == 2048:
            continue
        make_set(rows, raw_dir, gen / f"size_{s}", SIZE_SET_COUNT, s, seed=s)

    print("\nDone.")


if __name__ == "__main__":
    main()