#!/usr/bin/env python3
"""Aggregate both benchmark runs into per-(M-regime, N*K) median speedup for the
line-graph dashboard.

  novel_nk : speedup = default_us / o_us   (Origami speedup over default; >1 Origami faster)
  tuned    : speedup = o_us / csv_us       (tuned-config speedup over Origami; >1 tuned faster)
"""
import csv, json, statistics as st
from collections import defaultdict

NOVEL = "/home/demantri/origami_bench/results/estimation_vs_measured_novel_nk.csv"
TUNED = "/home/demantri/origami_bench/results/estimation_vs_measured_tuned.csv"
OUT = "/home/demantri/origami_bench/analysis/data/viz_line.json"

REGIMES = ["<64", "64-256", "256-1024", "1024-4096", ">4096"]

def regime(m):
    if m < 64: return "<64"
    if m <= 256: return "64-256"
    if m <= 1024: return "256-1024"
    if m <= 4096: return "1024-4096"
    return ">4096"

def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if (v == v and v not in (float("inf"), float("-inf"))) else None

def aggregate(path, numer, denom):
    cells = defaultdict(list)   # (regime, NK) -> [speeds]
    nklabel = {}
    for r in csv.DictReader(open(path)):
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        a, b = num(r.get(numer)), num(r.get(denom))
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        NK = N * K
        cells[(regime(M), NK)].append(a / b)
        nklabel[NK] = f"{N}x{K}"
    series = {reg: [] for reg in REGIMES}
    nkset = set()
    for (reg, NK), vals in cells.items():
        series[reg].append({"NK": NK, "nk": nklabel[NK],
                            "y": round(st.median(vals), 4), "n": len(vals)})
        nkset.add(NK)
    for reg in series:
        series[reg].sort(key=lambda p: p["NK"])
    nks = [{"NK": nk, "nk": nklabel[nk]} for nk in sorted(nkset)]
    return {"series": series, "nks": nks}

novel = aggregate(NOVEL, "default_us", "o_us")
tuned = aggregate(TUNED, "o_us", "csv_us")

data = {
    "regimes": REGIMES,
    "runs": [
        {
            "key": "novel",
            "title": "Origami vs default CK kernel - untuned (novel N,K) shapes",
            "ylabel": "Origami speedup over default  (default_us / origami_us)",
            "note": ">1.0x = Origami's pick is faster than aiter's default/backup CK kernel. "
                    "Source: estimation_vs_measured_novel_nk.csv (splitK=0).",
            **novel,
        },
        {
            "key": "tuned",
            "title": "Tuned CSV vs Origami - tuned shapes",
            "ylabel": "Tuned-config speedup over Origami  (origami_us / csv_us)",
            "note": ">1.0x = the tuned CSV kernel is faster than Origami's pick. "
                    "Source: estimation_vs_measured_tuned.csv (splitK=0; csv kernel measured at splitK=0).",
            **tuned,
        },
    ],
}
json.dump(data, open(OUT, "w"))
# quick console summary
for run in data["runs"]:
    print(f"\n[{run['key']}] {run['title']}")
    print(f"  distinct N*K: {len(run['nks'])}  ({run['nks'][0]['nk']} .. {run['nks'][-1]['nk']})")
    for reg in REGIMES:
        pts = run["series"][reg]
        ys = [p["y"] for p in pts]
        if ys:
            print(f"    {reg:>11}: {len(pts)} pts, median-speedup range {min(ys):.2f}..{max(ys):.2f}x, "
                  f"total n={sum(p['n'] for p in pts)}")
        else:
            print(f"    {reg:>11}: (no data)")
print(f"\nwrote {OUT}")
