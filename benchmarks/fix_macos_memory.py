#!/usr/bin/env python3
"""
ONE-OFF repair for results produced by the OLD binary on macOS (peak_mb was bytes/1024, i.e. KB
labelled as MB). Divides peak_mb by 1024 in benchmarks/results/*.csv.  Run it ONCE, BEFORE
rebuilding with the fixed main.cpp. A marker file stops it from running twice.
"""
from pathlib import Path
import pandas as pd
RES = Path(__file__).resolve().parent / "results"
marker = RES / ".memory_fixed"
if marker.exists(): raise SystemExit("already applied - nothing to do")
for f in sorted(RES.glob("E*.csv")):
    d = pd.read_csv(f); d["peak_mb"] = d["peak_mb"] / 1024.0; d.to_csv(f, index=False)
    print(f"fixed {f.name}: peak_mb now {d['peak_mb'].min():.1f} .. {d['peak_mb'].max():.1f} MB")
marker.write_text("peak_mb divided by 1024 once\n")
