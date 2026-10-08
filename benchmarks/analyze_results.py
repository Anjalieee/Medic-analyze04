#!/usr/bin/env python3
"""
Step 8b - extract the numbers you need for the report analysis (printed + saved to
results/tables/analysis_summary.md). Everything is computed from benchmarks/results/*.csv.

Metrics
  speedup S = T_seq / T_par            efficiency E = S / p
  Karp-Flatt serial fraction  e = (1/S - 1/p) / (1 - 1/p)
      -> if e stays constant as p grows, the limit is the serial part (Amdahl);
         if e keeps growing, overheads (memory bandwidth, scheduling, hyper-threads) are the cause.
  Amdahl estimate from the stage breakdown: serial fraction f = (load + roi) / total of the sequential run.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "results" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
STAGES = ["load", "resize", "denoise", "enhance", "segment", "roi", "features"]
lines = []

def say(s=""):
    print(s); lines.append(s)

def agg(df):
    return df.groupby(["mode", "threads", "images", "size"], as_index=False).agg(
        wall=("wall_s", "mean"), sd=("wall_s", "std"), ips=("img_per_s", "mean"), mem=("peak_mb", "mean"),
        **{f"s_{s}": (f"sum_{s}_ms", "mean") for s in STAGES}).fillna({"sd": 0})

def loglog_slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])

def e1():
    f = RES / "E1_threads.csv"
    if not f.exists(): return
    g = agg(pd.read_csv(f))
    seq = g[g["mode"] == "seq"].iloc[0]; tseq = seq["wall"]
    tot = sum(seq[f"s_{s}"] for s in STAGES)
    say("## E1 - thread scaling")
    say(f"Dataset: {int(seq['images'])} images, {int(seq['size'])}px. Sequential time: {tseq:.3f} s "
        f"({seq['ips']:.1f} images/s), peak memory {seq['mem']:.0f} MB.\n")
    say("Sequential stage share: " + ", ".join(f"{s} {100*seq[f's_{s}']/tot:.0f}%" for s in STAGES))
    f_ser = (seq["s_load"] + seq["s_roi"]) / tot
    say(f"Estimated serial fraction of the within-image mode (load + ROI labelling): {f_ser:.1%} "
        f"-> Amdahl upper bound on speedup = {1/f_ser:.1f}x\n")
    for m in ("omp_images", "omp_pixels"):
        d = g[g["mode"] == m].sort_values("threads").copy()
        d["S"] = tseq / d["wall"]; d["E"] = d["S"] / d["threads"]
        d["KF"] = [(1 / s - 1 / p) / (1 - 1 / p) if p > 1 else np.nan for s, p in zip(d["S"], d["threads"])]
        b = d.loc[d["S"].idxmax()]
        say(f"### {m}")
        say("| threads | time (s) | speedup | efficiency | Karp-Flatt e |\n|---|---|---|---|---|")
        for _, r in d.iterrows():
            say(f"| {int(r.threads)} | {r.wall:.3f} | {r.S:.2f} | {r.E:.2f} | {'-' if np.isnan(r.KF) else f'{r.KF:.3f}'} |")
        say(f"\nBest speedup {b.S:.2f}x at {int(b.threads)} threads (efficiency {b.E:.0%}); "
            f"at {int(d.threads.max())} threads: {d.iloc[-1].S:.2f}x, efficiency {d.iloc[-1].E:.0%}.")
        low = d[d["E"] < 0.7]
        if len(low): say(f"Efficiency first drops below 70% at {int(low.iloc[0].threads)} threads.")
        gain = d["S"].diff() / d["S"].shift()
        say("Marginal gain when adding threads: " + ", ".join(
            f"{int(p0)}->{int(p1)}: {100*gn:+.0f}%" for p0, p1, gn in zip(d.threads.shift(), d.threads, gain) if not np.isnan(gn)) + "\n")
        if m == "omp_pixels":
            one = d[d.threads == 1]
            if len(one): say(f"Overhead check: omp_pixels with 1 thread is {one.iloc[0].wall / tseq:.2f}x the sequential time "
                             f"(1.00 = no OpenMP overhead).\n")

def e2():
    f = RES / "E2_dataset.csv"
    if not f.exists(): return
    g = agg(pd.read_csv(f))
    say("## E2 - dataset-size scaling")
    say("| images | seq (s) | omp_images (s) | S | omp_pixels (s) | S | seq img/s | omp_images img/s | peak MB (seq) | peak MB (omp_images) |\n|---|---|---|---|---|---|---|---|---|---|")
    sizes = sorted(g["images"].unique())
    for n in sizes:
        r = {m: g[(g["mode"] == m) & (g["images"] == n)].iloc[0] for m in ("seq", "omp_images", "omp_pixels")}
        say(f"| {int(n)} | {r['seq'].wall:.2f} | {r['omp_images'].wall:.2f} | {r['seq'].wall/r['omp_images'].wall:.2f} | "
            f"{r['omp_pixels'].wall:.2f} | {r['seq'].wall/r['omp_pixels'].wall:.2f} | {r['seq'].ips:.1f} | "
            f"{r['omp_images'].ips:.1f} | {r['seq'].mem:.0f} | {r['omp_images'].mem:.0f} |")
    if len(sizes) > 2:
        for m in ("seq", "omp_images", "omp_pixels"):
            d = g[g["mode"] == m].sort_values("images")
            say(f"\n{m}: time ~ N^{loglog_slope(d.images, d.wall):.2f} (1.00 = perfectly linear in number of images)")
    say()

def e3():
    f = RES / "E3_imgsize.csv"
    if not f.exists(): return
    g = agg(pd.read_csv(f))
    say("## E3 - image-size scaling")
    say("| side (px) | seq (s) | omp_images (s) | S | omp_pixels (s) | S | peak MB (seq) | peak MB (omp_images) |\n|---|---|---|---|---|---|---|---|")
    sizes = sorted(g["size"].unique())
    for s in sizes:
        r = {m: g[(g["mode"] == m) & (g["size"] == s)].iloc[0] for m in ("seq", "omp_images", "omp_pixels")}
        say(f"| {int(s)} | {r['seq'].wall:.2f} | {r['omp_images'].wall:.2f} | {r['seq'].wall/r['omp_images'].wall:.2f} | "
            f"{r['omp_pixels'].wall:.2f} | {r['seq'].wall/r['omp_pixels'].wall:.2f} | {r['seq'].mem:.0f} | {r['omp_images'].mem:.0f} |")
    if len(sizes) > 2:
        for m in ("seq", "omp_images", "omp_pixels"):
            d = g[g["mode"] == m].sort_values("size")
            say(f"\n{m}: time ~ (side)^{loglog_slope(d['size'], d.wall):.2f}  (2.00 = linear in pixel count)")
    say()

if __name__ == "__main__":
    say("# Auto-generated results summary (verify against the graphs and rewrite in your own words)\n")
    e1(); e2(); e3()
    (OUT / "analysis_summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {OUT / 'analysis_summary.md'}")
