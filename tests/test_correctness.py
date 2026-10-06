#!/usr/bin/env python3
"""
Step 5 - correctness tests: sequential vs OpenMP.

 T1  seq vs omp_images  (1,2,4,8 threads): every non-timing CSV column identical
 T2  seq vs omp_pixels  (1,2,4,8 threads): every non-timing CSV column identical
 T3  repeatability: omp_images/omp_pixels run 3x with 4 threads give identical output
 T4  pixel-exact comparison of saved PNGs (enhanced image, mask, overlay) seq vs omp
 T5  known-answer test: synthetic bright square -> ROI bounding box found where expected
 T6  odd image sizes / tiny images do not crash and still match

Usage:
  python3 tests/test_correctness.py                       # synthetic images only
  python3 tests/test_correctness.py --images data/raw --labels data/labels.csv --split test --limit 200
"""
import argparse, csv, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SEQ, OMP = ROOT / "bin" / "mia_seq", ROOT / "bin" / "mia_omp"
THREADS = [1, 2, 4, 8]
results = []

def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))

def make_synthetic(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    sizes = [(512, 512), (640, 480), (333, 777), (1000, 800), (257, 255), (1500, 1200), (64, 64), (100, 37)]
    for i, (w, h) in enumerate(sizes):
        img = rng.normal(70, 12, (h, w)).clip(0, 255)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(i % 4):                                   # 0..3 bright blobs
            cx, cy, r = rng.integers(w // 5, 4 * w // 5), rng.integers(h // 5, 4 * h // 5), max(4, min(w, h) // 8)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = 210
        name = f"syn_{i:02d}.png"
        Image.fromarray(img.astype(np.uint8)).save(d / name)
        rows.append((name, 1 if i % 4 else 0, "test"))
    # known-answer image: one bright square on a dark background (512x512, no resize needed)
    img = rng.normal(40, 5, (512, 512)).clip(0, 255)
    img[150:270, 200:320] = 230
    Image.fromarray(img.astype(np.uint8)).save(d / "known_square.png")
    rows.append(("known_square.png", 1, "test"))
    with open(d / "labels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "label", "split"]); w.writerows(rows)

def run(binary, mode, threads, images, labels, split, limit, out_csv, save_dir=None):
    cmd = [str(binary), "--images", str(images), "--labels", str(labels), "--mode", mode,
           "--threads", str(threads), "--out", str(out_csv)]
    if split: cmd += ["--split", split]
    if limit: cmd += ["--limit", str(limit)]
    if save_dir: cmd += ["--save-dir", str(save_dir)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout, p.stderr)
    return p.returncode == 0

def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        cols = [c for c in r.fieldnames if not c.startswith("t_")]
        return [{c: row[c] for c in cols} for row in r]

def same(a, b):
    ra, rb = load(a), load(b)
    if len(ra) != len(rb): return False, "different number of rows"
    for x, y in zip(ra, rb):
        for k in x:
            if x[k] != y[k]: return False, f"{x['file']} column '{k}': {x[k]} != {y[k]}"
    return True, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--labels"); ap.add_argument("--split")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    for b in (SEQ, OMP):
        if not b.exists(): sys.exit(f"{b} missing - run `make` first")

    tmp = Path(tempfile.mkdtemp(prefix="mia_test_"))
    syn = ROOT / "tests" / "test_cases" / "synthetic"
    make_synthetic(syn)
    sets = [("synthetic", syn, syn / "labels.csv", None, None)]
    if a.images:
        sets.append(("real", Path(a.images), Path(a.labels or Path(a.images) / "labels.csv"), a.split, a.limit))

    for tag, img_dir, lab, split, limit in sets:
        ref = tmp / f"{tag}_seq.csv"
        ok = run(SEQ, "seq", 1, img_dir, lab, split, limit, ref)
        check(f"[{tag}] sequential baseline runs", ok)
        if not ok: continue
        for mode, tname in (("omp_images", "T1"), ("omp_pixels", "T2")):
            for t in THREADS:
                out = tmp / f"{tag}_{mode}_{t}.csv"
                ok = run(OMP, mode, t, img_dir, lab, split, limit, out)
                eq, why = same(ref, out) if ok else (False, "run failed")
                check(f"[{tag}] {tname} seq == {mode} @ {t} threads", ok and eq, why)
        for mode in ("omp_images", "omp_pixels"):
            outs = []
            for k in range(3):
                o = tmp / f"{tag}_rep_{mode}_{k}.csv"
                run(OMP, mode, 4, img_dir, lab, split, limit, o); outs.append(o)
            good = all(same(outs[0], o)[0] for o in outs[1:])
            check(f"[{tag}] T3 {mode} repeatable over 3 runs (4 threads)", good)

    # T4: pixel-exact PNG comparison on synthetic set
    sd = {m: tmp / f"png_{m}" for m in ("seq", "omp_images", "omp_pixels")}
    for d in sd.values(): d.mkdir()
    run(SEQ, "seq", 1, syn, syn / "labels.csv", None, None, tmp / "p0.csv", sd["seq"])
    for m in ("omp_images", "omp_pixels"):
        run(OMP, m, 4, syn, syn / "labels.csv", None, None, tmp / f"p_{m}.csv", sd[m])
        diffs = 0; n = 0
        for f in sorted(sd["seq"].glob("*.png")):
            x = np.array(Image.open(f)); y = np.array(Image.open(sd[m] / f.name))
            n += 1; diffs += int(x.shape != y.shape or np.any(x != y))
        check(f"T4 pixel-exact PNG outputs seq == {m} ({n} files)", n > 0 and diffs == 0, f"{diffs} files differ")

    # T5: known-answer test
    ref = load(tmp / "synthetic_seq.csv")
    row = next(r for r in ref if r["file"] == "known_square.png")
    boxes = [tuple(map(int, b.split(":"))) for b in row["rois"].split("|") if b]
    expect = (200, 150, 319, 269)                          # x0,y0,x1,y1 of the bright square
    hit = any(all(abs(b[i] - expect[i]) <= 4 for i in range(4)) for b in boxes)
    check("T5 known square detected as ROI within 4 px", hit, f"got {boxes}, expected ~{expect}")

    # T6 is covered by the 64x64 and 100x37 synthetic images inside T1-T3.
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
