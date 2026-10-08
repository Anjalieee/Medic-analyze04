#!/usr/bin/env python3
"""
Step 7a - run all performance experiments and write CSVs to benchmarks/results/.

  E1  thread scaling   : fixed dataset, seq vs omp_images vs omp_pixels, threads 1..max
  E2  dataset scaling  : count_100 ... count_10000 (512x512), seq vs omp (max threads)
  E3  image-size scale : size_256 ... size_2048 (200 images each), seq vs omp (max threads)

Every configuration is run --repeats times after one discarded warm-up run (so the OS file
cache is hot and we measure compute, not first-read disk speed). --no-hash is always used.

Usage:
  python3 benchmarks/run_benchmarks.py                 # full run
  python3 benchmarks/run_benchmarks.py --quick         # small smoke test
  python3 benchmarks/run_benchmarks.py --only E1       # one experiment
Close other heavy programs while benchmarking.
"""
import argparse, datetime, os, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ, OMP = ROOT / "bin" / "mia_seq", ROOT / "bin" / "mia_omp"
RES = ROOT / "benchmarks" / "results"

def thread_list(maxt):
    t, out = 1, []
    while t < maxt: out.append(t); t *= 2
    out.append(maxt)
    return sorted(set(out))

def run(mode, threads, ds_dir, size, summary, repeats, warmup=True):
    binary = SEQ if mode == "seq" else OMP
    cmd = [str(binary), "--images", str(ds_dir), "--mode", mode, "--threads", str(threads),
           "--size", str(size), "--no-hash"]
    for r in range(repeats + (1 if warmup else 0)):
        use_summary = not (warmup and r == 0)             # first run = warm-up, not recorded
        c = cmd + (["--summary", str(summary)] if use_summary else [])
        p = subprocess.run(c, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout, p.stderr); sys.exit(f"run failed: {' '.join(c)}")
    print(f"    done {mode:10s} threads={threads:<3d} {ds_dir.name} size={size}")

def fresh(path, append):
    if path.exists() and not append: path.unlink()

def sysinfo(maxt):
    cpu = platform.processor()
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"): cpu = l.split(":", 1)[1].strip(); break
    except OSError: pass
    comp = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    mem = ""
    try: mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].strip()
    except Exception: pass
    (RES / "system_info.txt").write_text(
        f"date: {datetime.datetime.now():%Y-%m-%d %H:%M}\ncpu: {cpu}\nlogical cores: {os.cpu_count()}\n"
        f"max threads used: {maxt}\n{mem}\ncompiler: {comp}\nflags: -O2 -ffp-contract=off -fopenmp\n"
        f"os: {platform.platform()}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--max-threads", type=int, default=os.cpu_count())
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--e1", default="count_1000", help="dataset folder used for the thread-scaling study")
    ap.add_argument("--counts", type=int, nargs="*", default=[100, 500, 1000, 5000, 10000])
    ap.add_argument("--sizes", type=int, nargs="*", default=[256, 512, 1024, 2048])
    ap.add_argument("--only", choices=["E1", "E2", "E3"])
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.repeats = 2
    data = Path(a.data); RES.mkdir(parents=True, exist_ok=True)
    T = thread_list(a.max_threads); mx = a.max_threads
    sysinfo(mx)
    print(f"threads tested: {T}   repeats: {a.repeats}")

    if a.only in (None, "E1"):
        print("E1: thread scaling"); f = RES / "E1_threads.csv"; fresh(f, a.append)
        d = data / a.e1
        run("seq", 1, d, 512, f, a.repeats)
        for mode in ("omp_images", "omp_pixels"):
            for t in T: run(mode, t, d, 512, f, a.repeats)

    if a.only in (None, "E2"):
        print("E2: dataset-size scaling"); f = RES / "E2_dataset.csv"; fresh(f, a.append)
        for n in a.counts:
            d = data / f"count_{n}"
            if not d.exists(): print(f"    skip {d} (missing)"); continue
            reps = a.repeats if n <= 1000 else max(1, a.repeats - 1)     # big sets: fewer repeats
            run("seq", 1, d, 512, f, reps)
            for mode in ("omp_images", "omp_pixels"): run(mode, mx, d, 512, f, reps)

    if a.only in (None, "E3"):
        print("E3: image-size scaling"); f = RES / "E3_imgsize.csv"; fresh(f, a.append)
        for s in a.sizes:
            d = data / f"size_{s}"
            if not d.exists(): print(f"    skip {d} (missing)"); continue
            run("seq", 1, d, s, f, a.repeats)
            for mode in ("omp_images", "omp_pixels"): run(mode, mx, d, s, f, a.repeats)
    print("\nFinished. Next: python3 benchmarks/plot_results.py")

if __name__ == "__main__":
    main()
