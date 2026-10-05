"""
prepare_data.py
Step 2a: Merge the Kaggle chest_xray train/val/test folders, create our own
stratified split, copy images to data/raw/ and write data/labels.csv.

Usage:
    python prepare_data.py --src path/to/chest_xray --dst data
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image

EXTS = {".jpeg", ".jpg", ".png"}


def collect(src: Path):
    """Find every image under src, label from parent folder name."""
    items = []
    for p in src.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        if "__MACOSX" in p.parts or p.name.startswith("."):
            continue
        parent = p.parent.name.upper()
        if parent not in ("NORMAL", "PNEUMONIA"):
            continue
        name = p.name.lower()
        if parent == "NORMAL":
            subtype = "normal"
        elif "bacteria" in name:
            subtype = "bacteria"
        elif "virus" in name:
            subtype = "virus"
        else:
            subtype = "pneumonia"
        items.append((p, 0 if parent == "NORMAL" else 1, subtype))
    return items


def stratified_split(items, ratios=(0.70, 0.15, 0.15), seed=42):
    rng = random.Random(seed)
    out = {}
    for label in (0, 1):
        group = [it for it in items if it[1] == label]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, it in enumerate(group):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            out[it[0]] = split
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", default=Path("data"), type=Path)
    args = ap.parse_args()

    items = collect(args.src)
    if not items:
        raise SystemExit("No images found. Check --src path.")
    splits = stratified_split(items)

    raw_dir = args.dst / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()
    for path, label, subtype in items:
        fname = path.name
        if fname in seen:  # avoid name clashes between folders
            fname = f"{path.parent.parent.name}_{fname}"
        seen.add(fname)
        try:
            with Image.open(path) as im:
                w, h = im.size
                mode = im.mode
        except Exception as e:
            print(f"Skipping corrupt file {path}: {e}")
            continue
        shutil.copy2(path, raw_dir / fname)
        rows.append({
            "filename": fname,
            "label": label,
            "label_name": "NORMAL" if label == 0 else "ABNORMAL",
            "subtype": subtype,
            "split": splits[path],
            "width": w,
            "height": h,
            "mode": mode,
        })

    csv_path = args.dst / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # Summary for the report
    print(f"\nWrote {len(rows)} images -> {raw_dir}")
    print(f"Wrote labels -> {csv_path}\n")
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        n0 = sum(1 for r in sub if r["label"] == 0)
        print(f"{split:5s}: {len(sub):5d} images  (normal {n0}, abnormal {len(sub) - n0})")
    ws = [r["width"] for r in rows]
    hs = [r["height"] for r in rows]
    print(f"\nWidth  min/mean/max: {min(ws)}/{sum(ws)//len(ws)}/{max(ws)}")
    print(f"Height min/mean/max: {min(hs)}/{sum(hs)//len(hs)}/{max(hs)}")


if __name__ == "__main__":
    main()