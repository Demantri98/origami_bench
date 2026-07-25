#!/usr/bin/env python3
"""Aggregate estimation_vs_measured_tuned.csv into per-(M-bucket, N, K) ratios
relative to the CSV (tuned) pick, for visualization."""
import csv, json, statistics as st
from collections import defaultdict

SRC = "/home/demantri/origami_bench/results/estimation_vs_measured_tuned.csv"
BUCKETS = [(0,16),(16,64),(64,256),(256,1024),(1024,4096),(4096,10**12)]
BLABEL = {(0,16):"≤16",(16,64):"16-64",(64,256):"64-256",
          (256,1024):"256-1024",(1024,4096):"1024-4096",(4096,10**12):">4096"}

def bof(m):
    for lo,hi in BUCKETS:
        if lo<m<=hi: return (lo,hi)

rows=[]
for r in csv.DictReader(open(SRC)):
    try:
        M,N,K=int(r["M"]),int(r["N"]),int(r["K"])
        o=float(r["o_us"]); c=float(r["csv_us"]); d=float(r["default_us"]); orc=float(r["oracle_us"])
    except Exception:
        continue
    if not (o>0 and c>0 and d>0 and orc>0): continue
    rows.append((M,N,K,o,c,d,orc))

cells=defaultdict(list)   # (blabel, NxK) -> list of dict ratios
for (M,N,K,o,c,d,orc) in rows:
    b=BLABEL[bof(M)]; nk=f"{N}x{K}"
    cells[(b,nk)].append(dict(o_vs_csv=o/c, def_vs_csv=d/c, orc_vs_csv=orc/c,
                              o_us=o, csv_us=c))

def med(xs): return round(st.median(xs),4)
def mean(xs): return round(st.mean(xs),4)

out=[]
for (b,nk),lst in cells.items():
    N,K=nk.split("x")
    out.append(dict(
        bucket=b, nk=nk, N=int(N), K=int(K), n=len(lst),
        o_vs_csv_med=med([x["o_vs_csv"] for x in lst]),
        o_vs_csv_mean=mean([x["o_vs_csv"] for x in lst]),
        def_vs_csv_med=med([x["def_vs_csv"] for x in lst]),
        def_vs_csv_mean=mean([x["def_vs_csv"] for x in lst]),
        orc_vs_csv_med=med([x["orc_vs_csv"] for x in lst]),
        # fraction where origami beats or ties csv (ratio<=1.001)
        o_wins_frac=round(sum(1 for x in lst if x["o_vs_csv"]<=1.001)/len(lst),3),
    ))

order_b={v:i for i,v in enumerate(BLABEL.values())}
out.sort(key=lambda r:(order_b[r["bucket"]], r["K"], r["N"]))
meta=dict(total_shapes=len(rows), buckets=list(BLABEL.values()),
          nks=sorted({r["nk"] for r in out}, key=lambda s:(int(s.split("x")[1]),int(s.split("x")[0]))))
print(json.dumps(dict(meta=meta, cells=out), indent=None))
