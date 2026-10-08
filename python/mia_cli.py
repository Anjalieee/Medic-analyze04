#!/usr/bin/env python3
"""
Medical image analysis - terminal interface (replaces the dashboard).

  organize  SRC            upload/organize images into data/uploads/<batch>/ with labels + statistics
  analyze                  run the pipeline: per-image results, detected regions (ASCII preview + saved
                           overlays), classification, accuracy metrics, time and memory
  compare                  sequential vs OpenMP (both modes): correctness check, time, speedup,
                           efficiency, memory, optional thread sweep
  scale                    live scalability test: increasing number of images / image sizes
  report                   show stored benchmark + classifier results (no re-running)
  demo                     analyze + compare + report in one go (good for the presentation)

Examples
  python3 python/mia_cli.py organize ~/Downloads/new_xrays --name batch1
  python3 python/mia_cli.py analyze --limit 15 --ascii 2
  python3 python/mia_cli.py analyze --images data/uploads/batch1
  python3 python/mia_cli.py compare --limit 300 --sweep
  python3 python/mia_cli.py scale --counts 100 500 1000 --sizes 256 512 1024
  python3 python/mia_cli.py report

NOTE: research/teaching prototype. It assists a reader and is NOT a medical diagnosis.
"""
import argparse, csv, os, platform, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SEQ, OMP = ROOT / "bin" / "mia_seq", ROOT / "bin" / "mia_omp"
MODEL = ROOT / "models" / "model.txt"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
STAGES = ["load", "resize", "denoise", "enhance", "segment", "roi", "features"]
NAMES = {0: "NORMAL", 1: "ABNORMAL"}
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
DISCLAIMER = ("DISCLAIMER: this tool is a research/teaching prototype that highlights regions for a human to\n"
              "review. It is NOT a medical device and must not be used for diagnosis or treatment decisions.")

# ------------------------------------------------------------------ terminal helpers
def col(s, code):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else str(s)
bold = lambda s: col(s, "1"); dim = lambda s: col(s, "2")
red = lambda s: col(s, "31"); green = lambda s: col(s, "32"); yellow = lambda s: col(s, "33"); cyan = lambda s: col(s, "36")

def title(t):
    print("\n" + bold(cyan(f"=== {t} " + "=" * max(0, 74 - len(t)))))

def visible_len(s):
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", str(s)))

def table(headers, rows, right=()):
    w = [max(visible_len(h), *(visible_len(r[i]) for r in rows)) if rows else visible_len(h) for i, h in enumerate(headers)]
    def line(cells):
        out = []
        for i, c in enumerate(cells):
            pad = w[i] - visible_len(c)
            out.append((" " * pad + str(c)) if i in right else (str(c) + " " * pad))
        return "│ " + " │ ".join(out) + " │"
    sep = lambda l, m, r: l + m.join("─" * (x + 2) for x in w) + r
    print(sep("┌", "┬", "┐")); print(line([bold(h) for h in headers])); print(sep("├", "┼", "┤"))
    for r in rows: print(line(r))
    print(sep("└", "┴", "┘"))

def bar(v, vmax, width=28, ch="█"):
    n = 0 if vmax <= 0 else int(round(width * min(v, vmax) / vmax))
    return ch * n + dim("·" * (width - n))

def need_bins(*paths):
    for p in paths:
        if not p.exists(): sys.exit(f"{p} not found - run `make` in the project root first.")

# ------------------------------------------------------------------ input preparation
def find_images(d):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT and "__MACOSX" not in p.parts and not p.name.startswith("."))

def guess_label(p):
    n = p.parent.name.lower()
    if "normal" in n and "ab" not in n: return 0
    if any(k in n for k in ("pneumonia", "abnormal", "tumor", "fracture")): return 1
    return -1

class Inp:  # what the C++ binary needs
    def __init__(self, images, labels, split, note): self.images, self.labels, self.split, self.note = images, labels, split, note

def prepare(images, labels, split, tmp):
    p = Path(images)
    if p.is_file():
        d = Path(tmp) / "single"; d.mkdir(exist_ok=True)
        with open(d / "labels.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["filename", "label"]); w.writerow([p.name, -1])
        return Inp(str(p.parent), str(d / "labels.csv"), None, f"single file {p.name} (no ground truth)")
    if not p.is_dir(): sys.exit(f"not found: {p}")
    lab = Path(labels) if labels else None
    if lab is None and (p / "labels.csv").exists(): lab = p / "labels.csv"
    if lab is None and p.name == "raw" and (p.parent / "labels.csv").exists():
        lab = p.parent / "labels.csv"; split = split or "test"
    if split == "all": split = None
    if lab is not None:
        return Inp(str(p), str(lab), split, f"labels: {lab}" + (f", split={split}" if split else ""))
    files = find_images(p)
    if not files: sys.exit(f"no images found in {p}")
    out = Path(tmp) / "auto_labels.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "label"])
        for q in files: w.writerow([q.relative_to(p).as_posix(), guess_label(q)])
    return Inp(str(p), str(out), None, "no labels.csv: labels guessed from folder names (NORMAL/PNEUMONIA), else unknown")

# ------------------------------------------------------------------ running the C++ binaries
def run_mia(mode, threads, inp, size=512, limit=None, out=None, save=None, hash=False, model=None):
    binary = SEQ if mode == "seq" else OMP
    summ = Path(tempfile.mkdtemp()) / "s.csv"
    cmd = [str(binary), "--images", inp.images, "--labels", inp.labels, "--mode", mode, "--threads", str(threads),
           "--size", str(size), "--summary", str(summ)]
    if inp.split: cmd += ["--split", inp.split]
    if limit: cmd += ["--limit", str(limit)]
    if out: cmd += ["--out", str(out)]
    if save: cmd += ["--save-dir", str(save)]
    if not hash: cmd += ["--no-hash"]
    if model: cmd += ["--model", str(model)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0, 2) or not summ.exists():
        sys.exit(f"pipeline failed:\n{p.stdout}\n{p.stderr}")
    if p.returncode == 2: print(yellow("warning: some images could not be read and were skipped"))
    row = list(csv.DictReader(open(summ)))[-1]
    return {k: (v if k == "mode" else float(v)) for k, v in row.items()}

def read_results(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["rois_l"] = [tuple(int(x) for x in q.split(":")) for q in r["rois"].split("|") if q]
    return rows

def same_outputs(a, b):
    ra, rb = list(csv.DictReader(open(a))), list(csv.DictReader(open(b)))
    if len(ra) != len(rb): return False
    return all(all(x[k] == y[k] for k in x if not k.startswith("t_")) for x, y in zip(ra, rb))

def thread_list(mx):
    t, out = 1, []
    while t < mx: out.append(t); t *= 2
    return sorted(set(out + [mx]))

def default_model(a):
    if a.model: return Path(a.model)
    return MODEL if MODEL.exists() else None

# ------------------------------------------------------------------ ASCII preview with boxes
RAMP = " .:-=+*#%@"
def ascii_preview(png, rois, cols=64, rows=30, max_boxes=5):
    im = Image.open(png).convert("L"); W, H = im.size
    a = np.asarray(im.resize((cols, rows), Image.BILINEAR))
    grid = [[(RAMP[int(v) * (len(RAMP) - 1) // 255], False) for v in r] for r in a]
    for (x0, y0, x1, y1, _) in sorted(rois, key=lambda r: -r[4])[:max_boxes]:
        c0, c1 = int(x0 * cols / W), min(cols - 1, int(x1 * cols / W))
        r0, r1 = int(y0 * rows / H), min(rows - 1, int(y1 * rows / H))
        for c in range(c0, c1 + 1): grid[r0][c] = ("-", True); grid[r1][c] = ("-", True)
        for r in range(r0, r1 + 1): grid[r][c0] = ("|", True); grid[r][c1] = ("|", True)
        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)): grid[r][c] = ("+", True)
    print("┌" + "─" * cols + "┐")
    for r in grid: print("│" + "".join(red(ch) if hit else ch for ch, hit in r) + "│")
    print("└" + "─" * cols + "┘")

# ------------------------------------------------------------------ commands
def cmd_organize(a):
    src = Path(a.src).expanduser()
    files = find_images(src) if src.is_dir() else [src]
    if not files: sys.exit("no images found")
    dst = ROOT / "data" / "uploads" / (a.name or "batch1"); dst.mkdir(parents=True, exist_ok=True)
    rows, bad, seen = [], [], set()
    for p in files:
        try:
            with Image.open(p) as im: w, h = im.size
        except Exception as e:
            bad.append((p.name, str(e)[:40])); continue
        name = p.name if p.name not in seen else f"{p.parent.name}_{p.name}"
        seen.add(name); shutil.copy2(p, dst / name)
        lab = {"normal": 0, "abnormal": 1}.get(a.label, guess_label(p))
        rows.append((name, lab, w, h, p.suffix.lower().lstrip(".")))
    with open(dst / "labels.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["filename", "label", "width", "height", "format"]); wr.writerows(rows)
    title("Organized upload")
    ws = np.array([r[2] for r in rows]); hs = np.array([r[3] for r in rows])
    fm = {}
    for r in rows: fm[r[4]] = fm.get(r[4], 0) + 1
    table(["item", "value"], [
        ["destination", str(dst.relative_to(ROOT))], ["images stored", len(rows)], ["unreadable (skipped)", len(bad)],
        ["labels", f"normal={sum(r[1]==0 for r in rows)}  abnormal={sum(r[1]==1 for r in rows)}  unknown={sum(r[1]==-1 for r in rows)}"],
        ["width  min/mean/max", f"{ws.min()} / {int(ws.mean())} / {ws.max()}"],
        ["height min/mean/max", f"{hs.min()} / {int(hs.mean())} / {hs.max()}"], ["formats", fm]])
    for n, e in bad: print(yellow(f"  skipped {n}: {e}"))
    print(f"\nNext: python3 python/mia_cli.py analyze --images {dst.relative_to(ROOT)}")

def cmd_analyze(a, quiet_header=False):
    need_bins(OMP)
    tmp = tempfile.mkdtemp(); inp = prepare(a.images, a.labels, a.split, tmp)
    model = default_model(a)
    save = ROOT / "results" / "output_images"; save.mkdir(parents=True, exist_ok=True)
    out = Path(tmp) / "res.csv"; T = a.threads or os.cpu_count()
    if not quiet_header:
        print(bold("AI-Based Medical Image Processing and Analysis System - terminal edition")); print(dim(DISCLAIMER))
    title("Run configuration")
    print(f"input   : {inp.images}\n{inp.note}")
    print(f"mode    : {a.mode}  threads={T if a.mode != 'seq' else 1}   resize={a.size}px")
    print(f"model   : {model if model else red('none - UNTRAINED placeholder weights (run train_classifier.py)')}")
    s = run_mia("seq" if a.mode == "seq" else a.mode, T, inp, a.size, a.limit, out, save, False, model)
    res = read_results(out)

    title(f"Per-image results (first {min(a.show, len(res))} of {len(res)})")
    rows = []
    for i, r in enumerate(res[:a.show], 1):
        pred, lab = int(r["pred"]), int(r["label"])
        pc = red(NAMES[pred]) if pred else green(NAMES[pred])
        big = max(r["rois_l"], key=lambda q: q[4], default=None)
        truth = dim("unknown") if lab < 0 else (NAMES[lab] + (green(" ✓") if lab == pred else red(" ✗")))
        rows.append([i, r["file"][:30], pc, f"{float(r['score']):+.2f}", r["n_rois"],
                     f"({big[0]},{big[1]})-({big[2]},{big[3]})" if big else "-", truth])
    table(["#", "image", "prediction", "score", "ROIs", "largest region", "ground truth"], rows, right=(0, 3, 4))
    print(dim("score > 0 => ABNORMAL. Regions are in the resized image's pixel coordinates."))

    for r in res[:a.ascii]:
        print("\n" + bold(f"Detected regions: {r['file']}  ->  {colored_pred(r)}") + dim(f"   (largest {5} boxes drawn in red)"))
        png = save / (r["file"].replace("/", "_") + "_enh.png")
        if png.exists(): ascii_preview(png, r["rois_l"])
    print(f"\nOverlay images (full resolution, boxes drawn): {save.relative_to(ROOT)}/*_overlay.png")
    if a.open and res:
        first = save / (res[0]["file"].replace("/", "_") + "_overlay.png")
        subprocess.run(["open" if platform.system() == "Darwin" else "xdg-open", str(first)])

    title("Classification summary")
    n_ab = sum(int(r["pred"]) for r in res)
    print(f"predicted: {green(str(len(res)-n_ab))} normal, {red(str(n_ab))} abnormal  (of {len(res)})")
    lab = [(int(r["label"]), int(r["pred"])) for r in res if int(r["label"]) in (0, 1)]
    if lab:
        tp = sum(l == 1 and p == 1 for l, p in lab); fn = sum(l == 1 and p == 0 for l, p in lab)
        tn = sum(l == 0 and p == 0 for l, p in lab); fp = sum(l == 0 and p == 1 for l, p in lab)
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0; sp = tn / (tn + fp) if tn + fp else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        base = max(tp + fn, tn + fp) / len(lab)
        print(f"\n               predicted NORMAL   predicted ABNORMAL\n  true NORMAL  {tn:>14}   {fp:>18}\n  true ABNORMAL{fn:>14}   {tp:>18}\n")
        table(["accuracy", "precision", "recall (sensitivity)", "specificity", "F1", "majority-class baseline"],
              [[f"{(tp+tn)/len(lab):.3f}", f"{pr:.3f}", f"{rc:.3f}", f"{sp:.3f}", f"{f1:.3f}", f"{base:.3f}"]])
        print(dim("Compare accuracy with the baseline (always guess the bigger class): accuracy alone can mislead on imbalanced data."))
    else:
        print(dim("No ground-truth labels for these images, so accuracy cannot be computed."))

    title("Performance")
    tot = sum(s[f"sum_{k}_ms"] for k in STAGES)
    print(f"wall time {s['wall_s']:.3f} s   throughput {s['img_per_s']:.1f} images/s   peak memory {s['peak_mb']:.1f} MB")
    print(dim("stage times are CPU time summed over images" + (" and threads" if a.mode == "omp_images" else "") + ":"))
    for k in STAGES:
        v = s[f"sum_{k}_ms"]; print(f"  {k:<9}{v:>10.1f} ms {100*v/tot:5.1f}%  {bar(v, tot)}")
    return res

def colored_pred(r):
    return red("ABNORMAL") if int(r["pred"]) else green("NORMAL")

def cmd_compare(a, quiet_header=False):
    need_bins(SEQ, OMP)
    tmp = tempfile.mkdtemp(); inp = prepare(a.images, a.labels, a.split, tmp)
    T = a.threads or os.cpu_count(); size = a.size
    if not quiet_header: print(bold("Sequential vs OpenMP comparison")); print(dim(DISCLAIMER))
    title("Setup")
    print(f"input: {inp.images}   {inp.note}\nlimit={a.limit or 'all'}  threads={T}  repeats={a.repeats}  resize={size}px")

    title("1. Correctness (outputs must be identical, incl. checksums of every intermediate image)")
    outs = {}
    for m in ("seq", "omp_images", "omp_pixels"):
        outs[m] = Path(tmp) / f"{m}.csv"; run_mia(m, 1 if m == "seq" else T, inp, size, a.limit, outs[m], hash=True)
    ok = [same_outputs(outs["seq"], outs[m]) for m in ("omp_images", "omp_pixels")]
    for m, k in zip(("omp_images", "omp_pixels"), ok):
        print(f"  seq == {m:<11} : {green('IDENTICAL ✓') if k else red('DIFFERENT ✗  (race condition / bug!)')}")

    title("2. Timing")
    run_mia("omp_images", T, inp, size, a.limit)                       # warm-up (file cache)
    def timed(m, t):
        rs = [run_mia(m, t, inp, size, a.limit) for _ in range(a.repeats)]
        return {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k != "mode"} | {"sd": float(np.std([r["wall_s"] for r in rs]))}
    base = timed("seq", 1); cfg = [("seq", 1, base)]
    tl = thread_list(T) if a.sweep else [T]
    for m in ("omp_images", "omp_pixels"):
        for t in tl: cfg.append((m, t, timed(m, t)))
    rows = []
    for m, t, r in cfg:
        sp = base["wall_s"] / r["wall_s"]
        rows.append([m, t, f"{r['wall_s']:.3f} ±{r['sd']:.3f}", f"{sp:.2f}x", f"{100*sp/t:.0f}%", f"{r['img_per_s']:.1f}", f"{r['peak_mb']:.0f}",
                     bar(sp, max(T, 1), 24)])
    table(["mode", "threads", "time (s)", "speedup", "efficiency", "images/s", "peak MB", f"speedup (full bar = ideal {T}x)"], rows, right=(1, 2, 3, 4, 5, 6))
    tot = sum(base[f"sum_{k}_ms"] for k in STAGES); f_ser = (base["sum_load_ms"] + base["sum_roi_ms"]) / tot
    print(f"\nAmdahl check: load + ROI labelling are sequential = {100*f_ser:.0f}% of the work -> within-image speedup cannot exceed {1/f_ser:.1f}x")
    print(dim("Speedup = T_seq / T_parallel,  efficiency = speedup / threads.  Times are means of the repeats (± std dev)."))

def sets_in(data, prefix):
    out = []
    for d in data.glob(f"{prefix}_*"):
        try: out.append((int(d.name.split("_")[1]), d))
        except ValueError: pass
    return sorted(out)

def cmd_scale(a, quiet_header=False):
    need_bins(SEQ, OMP)
    data = Path(a.data); T = a.threads or os.cpu_count()
    if not quiet_header: print(bold("Scalability test (live)")); print(dim(DISCLAIMER))
    def sweep(kind, wanted, size_of):
        found = [(n, d) for n, d in sets_in(data, kind) if not wanted or n in wanted]
        if not found: print(yellow(f"no {kind}_* folders in {data}")); return
        rows, first = [], None
        for n, d in found:
            inp = Inp(str(d), str(d / "labels.csv"), None, "")
            run_mia("omp_images", T, inp, size_of(n))                      # warm-up: hot file cache, not disk speed
            r = {m: run_mia(m, 1 if m == "seq" else T, inp, size_of(n)) for m in ("seq", "omp_images", "omp_pixels")}
            first = first or (n, r["seq"]["wall_s"])
            growth = r["seq"]["wall_s"] / first[1]; ratio = n / first[0] if kind == "count" else (n / first[0]) ** 2
            rows.append([n, f"{r['seq']['wall_s']:.2f}", f"{r['omp_images']['wall_s']:.2f}", f"{r['seq']['wall_s']/r['omp_images']['wall_s']:.2f}x",
                         f"{r['omp_pixels']['wall_s']:.2f}", f"{r['seq']['wall_s']/r['omp_pixels']['wall_s']:.2f}x",
                         f"{r['omp_images']['img_per_s']:.0f}", f"{r['seq']['peak_mb']:.0f}/{r['omp_images']['peak_mb']:.0f}",
                         f"{growth:.1f}x / {ratio:.0f}x"])
        table([("images" if kind == "count" else "side px"), "seq (s)", "omp_images (s)", "speedup", "omp_pixels (s)", "speedup",
               "omp_img img/s", "peak MB seq/omp", "time growth / work growth"], rows, right=tuple(range(9)))
    title(f"Increasing NUMBER of images (512px, {T} threads)")
    sweep("count", set(a.counts or []), lambda n: 512)
    title(f"Increasing IMAGE SIZE ({T} threads)")
    sweep("size", set(a.sizes or []), lambda n: n)
    print(dim("\n'time growth / work growth' close to 1:1 means the pipeline scales linearly with the amount of data."))

def cmd_report(a, quiet_header=False):
    import pandas as pd
    R = ROOT / "benchmarks" / "results"
    if not quiet_header: print(bold("Stored results report")); print(dim(DISCLAIMER))
    si = R / "system_info.txt"
    if si.exists(): title("System used for the benchmarks"); print(si.read_text().strip())
    cm = ROOT / "results" / "classifier_metrics.json"
    if cm.exists():
        import json
        m = json.loads(cm.read_text()); title("Classifier performance (held-out data)")
        rows = [[s, f"{m[s]['accuracy']:.3f}", f"{m[s]['precision']:.3f}", f"{m[s]['recall']:.3f}", f"{m[s]['specificity']:.3f}",
                 f"{m[s]['f1']:.3f}", f"{m[s]['baseline_all_abnormal_accuracy']:.3f}"] for s in ("val", "test") if s in m]
        table(["split", "accuracy", "precision", "recall", "specificity", "F1", "all-abnormal baseline"], rows)
    else:
        print(yellow("\nno results/classifier_metrics.json yet - run python/train_classifier.py"))
    def load(name):
        f = R / name
        if not f.exists(): return None
        d = pd.read_csv(f)
        return d.groupby(["mode", "threads", "images", "size"], as_index=False).agg(wall=("wall_s", "mean"), ips=("img_per_s", "mean"), mem=("peak_mb", "mean"))
    e1 = load("E1_threads.csv")
    if e1 is not None:
        seq = e1[e1["mode"] == "seq"].iloc[0]; title(f"E1 thread scaling ({int(seq.images)} images, {int(seq['size'])}px)")
        rows = []
        for m in ("seq", "omp_images", "omp_pixels"):
            for _, r in e1[e1["mode"] == m].sort_values("threads").iterrows():
                sp = seq.wall / r.wall; rows.append([m, int(r.threads), f"{r.wall:.3f}", f"{sp:.2f}x", f"{100*sp/r.threads:.0f}%", f"{r.ips:.1f}", bar(sp, e1.threads.max(), 24)])
        table(["mode", "threads", "time (s)", "speedup", "efficiency", "images/s", f"speedup (full = ideal {int(e1.threads.max())}x)"], rows, right=(1, 2, 3, 4, 5))
    for name, key, head in (("E2_dataset.csv", "images", "E2 increasing number of images"), ("E3_imgsize.csv", "size", "E3 increasing image size")):
        d = load(name)
        if d is None: continue
        title(head); rows = []
        for v in sorted(d[key].unique()):
            s = d[(d[key] == v) & (d["mode"] == "seq")].iloc[0]; i = d[(d[key] == v) & (d["mode"] == "omp_images")].iloc[0]; p = d[(d[key] == v) & (d["mode"] == "omp_pixels")].iloc[0]
            rows.append([int(v), f"{s.wall:.2f}", f"{i.wall:.2f}", f"{s.wall/i.wall:.2f}x", f"{p.wall:.2f}", f"{s.wall/p.wall:.2f}x", f"{s.mem:.0f}/{i.mem:.0f}"])
        table([("images" if key == "images" else "side px"), "seq (s)", "omp_images (s)", "speedup", "omp_pixels (s)", "speedup", "peak MB seq/omp"], rows, right=tuple(range(7)))
    if e1 is None: print(yellow("no benchmark CSVs found - run benchmarks/run_benchmarks.py"))

def cmd_demo(a):
    print(bold("AI-Based Medical Image Processing and Analysis System - demo")); print(dim(DISCLAIMER))
    a.show, a.ascii, a.limit = 12, 1, 40
    cmd_analyze(a, True)
    a.limit, a.sweep = 200, True
    cmd_compare(a, True)
    cmd_report(a, True)

# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description="Medical image analysis - terminal interface", epilog=DISCLAIMER)
    sp = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--images", default=str(ROOT / "data" / "raw"), help="folder or single image (default data/raw)")
        p.add_argument("--labels"); p.add_argument("--split", help="train|val|test|all (default: test for data/raw)")
        p.add_argument("--limit", type=int); p.add_argument("--threads", type=int); p.add_argument("--size", type=int, default=512)
    o = sp.add_parser("organize"); o.add_argument("src"); o.add_argument("--name"); o.add_argument("--label", choices=["normal", "abnormal"])
    an = sp.add_parser("analyze"); common(an)
    an.add_argument("--mode", default="omp_images", choices=["seq", "omp_images", "omp_pixels"]); an.add_argument("--model")
    an.add_argument("--show", type=int, default=15); an.add_argument("--ascii", type=int, default=0, help="ASCII preview of first K images")
    an.add_argument("--open", action="store_true", help="open the first overlay image in the system viewer")
    c = sp.add_parser("compare"); common(c); c.add_argument("--repeats", type=int, default=3); c.add_argument("--sweep", action="store_true")
    s = sp.add_parser("scale"); s.add_argument("--data", default=str(ROOT / "data" / "generated")); s.add_argument("--threads", type=int)
    s.add_argument("--counts", type=int, nargs="*"); s.add_argument("--sizes", type=int, nargs="*")
    sp.add_parser("report")
    d = sp.add_parser("demo"); common(d); d.add_argument("--mode", default="omp_images"); d.add_argument("--model")
    d.add_argument("--repeats", type=int, default=2); d.add_argument("--open", action="store_true")
    a = ap.parse_args()
    {"organize": cmd_organize, "analyze": cmd_analyze, "compare": cmd_compare, "scale": cmd_scale, "report": cmd_report, "demo": cmd_demo}[a.cmd](a)

if __name__ == "__main__":
    main()
