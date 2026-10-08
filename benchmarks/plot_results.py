#!/usr/bin/env python3
"""
Step 7b - turn benchmarks/results/*.csv into graphs (results/graphs/) and tables (results/tables/).

Speedup is always measured against the SEQUENTIAL binary (mia_seq) on the same dataset:
    speedup = T_seq / T_parallel      efficiency = speedup / threads
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmarks" / "results"
GR = ROOT / "results" / "graphs"; TB = ROOT / "results" / "tables"
GR.mkdir(parents=True, exist_ok=True); TB.mkdir(parents=True, exist_ok=True)
STAGES = ["load", "resize", "denoise", "enhance", "segment", "roi", "features"]
COL = {"seq": "black", "omp_images": "tab:blue", "omp_pixels": "tab:orange"}
LAB = {"seq": "sequential", "omp_images": "OpenMP - across images", "omp_pixels": "OpenMP - within image (pixels/rows)"}

def agg(df, keys):
    g = df.groupby(keys, as_index=False).agg(
        wall=("wall_s", "mean"), wall_std=("wall_s", "std"), runs=("wall_s", "count"),
        ips=("img_per_s", "mean"), mem=("peak_mb", "mean"),
        **{f"s_{s}": (f"sum_{s}_ms", "mean") for s in STAGES})
    g["wall_std"] = g["wall_std"].fillna(0)
    return g

def add_speedup(g, base_keys):
    base = g[g["mode"] == "seq"].set_index(base_keys)["wall"]
    g["t_seq"] = [base.get(tuple(r[k] for k in base_keys) if len(base_keys) > 1 else r[base_keys[0]], float("nan"))
                  for _, r in g.iterrows()]
    g["speedup"] = g["t_seq"] / g["wall"]
    g["efficiency"] = g["speedup"] / g["threads"]
    return g

def md_table(df, cols, path, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    Path(path).write_text("\n".join(lines) + "\n")

def save(fig, name):
    fig.tight_layout(); fig.savefig(GR / name, dpi=160); plt.close(fig); print("wrote", GR / name)

def e1():
    f = RES / "E1_threads.csv"
    if not f.exists(): return
    g = add_speedup(agg(pd.read_csv(f), ["mode", "threads", "images", "size"]), ["images", "size"])
    seq = g[g["mode"] == "seq"].iloc[0]
    par = g[g["mode"] != "seq"]; tmax = int(par["threads"].max())

    # Amdahl estimate for the within-image mode: load + ROI labelling are not parallelised
    tot = sum(seq[f"s_{s}"] for s in STAGES)
    serial = (seq["s_load"] + seq["s_roi"]) / tot
    xs = sorted(par["threads"].unique())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(seq["wall"], color=COL["seq"], ls="--", label=LAB["seq"])
    for m, d in par.groupby("mode"):
        ax.errorbar(d["threads"], d["wall"], d["wall_std"], marker="o", capsize=3, color=COL[m], label=LAB[m])
    ax.set_xlabel("threads"); ax.set_ylabel("wall time (s)"); ax.set_xscale("log", base=2)
    ax.set_title(f"Execution time vs threads ({int(seq['images'])} images, {int(seq['size'])}px)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, "e1_time_vs_threads.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, xs, "k:", label="ideal")
    for m, d in par.groupby("mode"): ax.plot(d["threads"], d["speedup"], "o-", color=COL[m], label=LAB[m])
    ax.plot(xs, [1 / (serial + (1 - serial) / p) for p in xs], "--", color=COL["omp_pixels"], alpha=.6,
            label=f"Amdahl estimate, within-image (serial={serial:.0%})")
    ax.set_xlabel("threads"); ax.set_ylabel("speedup  (T_seq / T_par)"); ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_title("Speedup vs threads"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    save(fig, "e1_speedup.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for m, d in par.groupby("mode"): ax.plot(d["threads"], d["efficiency"], "o-", color=COL[m], label=LAB[m])
    ax.axhline(1, color="k", ls=":"); ax.set_xlabel("threads"); ax.set_ylabel("parallel efficiency (speedup / threads)")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.15); ax.set_title("Parallel efficiency"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, "e1_efficiency.png")

    # per-stage breakdown: average ms per image
    rows = [("sequential", seq)]
    for m in ("omp_pixels", "omp_images"):
        r = par[(par["mode"] == m) & (par["threads"] == tmax)]
        if len(r): rows.append((f"{LAB[m].split(' - ')[1]}\n@{tmax}T", r.iloc[0]))
    fig, ax = plt.subplots(figsize=(7, 4)); bottom = [0] * len(rows)
    for s in STAGES:
        vals = [r[f"s_{s}"] / r["images"] for _, r in rows]
        ax.bar([n for n, _ in rows], vals, bottom=bottom, label=s); bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("avg ms per image (summed over threads for 'across images')"); ax.set_title("Stage-wise time breakdown")
    ax.legend(fontsize=7, ncol=2); save(fig, "e1_stage_breakdown.png")

    t = g.copy(); t["mode"] = t["mode"].map(LAB)
    md_table(t.sort_values(["mode", "threads"]), ["mode", "threads", "wall", "wall_std", "speedup", "efficiency", "ips"],
             TB / "e1_threads.md", {"wall": "{:.3f}", "wall_std": "{:.3f}", "speedup": "{:.2f}", "efficiency": "{:.2f}", "ips": "{:.1f}"})
    pd.DataFrame({"stage": STAGES, "pct_of_sequential_time": [100 * seq[f"s_{s}"] / tot for s in STAGES]}) \
        .to_csv(TB / "e1_stage_share.csv", index=False)
    print(f"serial fraction estimate (load+roi): {serial:.1%}  -> Amdahl max speedup {1/serial:.1f}x")

def e2():
    f = RES / "E2_dataset.csv"
    if not f.exists(): return
    g = add_speedup(agg(pd.read_csv(f), ["mode", "threads", "images", "size"]), ["images"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for m, d in g.groupby("mode"):
        axs[0].errorbar(d["images"], d["wall"], d["wall_std"], marker="o", capsize=3, color=COL[m], label=LAB[m])
        axs[1].plot(d["images"], d["ips"], "o-", color=COL[m], label=LAB[m])
        axs[2].plot(d["images"], d["mem"], "o-", color=COL[m], label=LAB[m])
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_title("Time vs number of images"); axs[0].set_ylabel("wall time (s)")
    axs[1].set_xscale("log"); axs[1].set_title("Throughput"); axs[1].set_ylabel("images / s")
    axs[2].set_xscale("log"); axs[2].set_title("Peak memory (RSS)"); axs[2].set_ylabel("MB")
    for a in axs: a.set_xlabel("number of images"); a.grid(alpha=.3)
    axs[0].legend(fontsize=7); save(fig, "e2_dataset_scaling.png")
    t = g[g["mode"] != "seq"].copy(); t["mode"] = t["mode"].map(LAB)
    md_table(t, ["mode", "images", "threads", "wall", "t_seq", "speedup", "efficiency", "ips", "mem"], TB / "e2_dataset.md",
             {"wall": "{:.3f}", "t_seq": "{:.3f}", "speedup": "{:.2f}", "efficiency": "{:.2f}", "ips": "{:.1f}", "mem": "{:.0f}"})

def e3():
    f = RES / "E3_imgsize.csv"
    if not f.exists(): return
    g = add_speedup(agg(pd.read_csv(f), ["mode", "threads", "images", "size"]), ["size"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for m, d in g.groupby("mode"):
        axs[0].errorbar(d["size"], d["wall"], d["wall_std"], marker="o", capsize=3, color=COL[m], label=LAB[m])
        if m != "seq": axs[1].plot(d["size"], d["speedup"], "o-", color=COL[m], label=LAB[m])
        axs[2].plot(d["size"], d["mem"], "o-", color=COL[m], label=LAB[m])
    axs[0].set_xscale("log", base=2); axs[0].set_yscale("log"); axs[0].set_title("Time vs image size"); axs[0].set_ylabel("wall time (s)")
    axs[1].set_xscale("log", base=2); axs[1].set_title("Speedup vs image size"); axs[1].set_ylabel("speedup")
    axs[2].set_xscale("log", base=2); axs[2].set_title("Peak memory (RSS)"); axs[2].set_ylabel("MB")
    for a in axs: a.set_xlabel("image side (pixels)"); a.grid(alpha=.3)
    axs[0].legend(fontsize=7); save(fig, "e3_image_size_scaling.png")
    t = g[g["mode"] != "seq"].copy(); t["mode"] = t["mode"].map(LAB)
    md_table(t, ["mode", "size", "threads", "wall", "t_seq", "speedup", "efficiency", "mem"], TB / "e3_imgsize.md",
             {"wall": "{:.3f}", "t_seq": "{:.3f}", "speedup": "{:.2f}", "efficiency": "{:.2f}", "mem": "{:.0f}"})

if __name__ == "__main__":
    e1(); e2(); e3()
