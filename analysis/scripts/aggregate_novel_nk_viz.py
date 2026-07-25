#!/usr/bin/env python3
"""Aggregate estimation_vs_measured_novel_nk.csv for the canvas:
Origami-vs-default speedup (+ vs-oracle ratios) per (M-bucket, N, K)."""
import csv, json, statistics as st
from collections import defaultdict

SRC = "/home/demantri/origami_bench/results/estimation_vs_measured_novel_nk.csv"
BUCKETS = [(0,16),(16,64),(64,256),(256,1024),(1024,4096),(4096,10**12)]
BLABEL = {(0,16):"≤16",(16,64):"16-64",(64,256):"64-256",(256,1024):"256-1024",
          (1024,4096):"1024-4096",(4096,10**12):">4096"}
def bof(m):
    for lo,hi in BUCKETS:
        if lo<m<=hi: return BLABEL[(lo,hi)]

def num(r,k):
    try: return float(r[k])
    except: return None

INF = float("inf")
rows=[]
for r in csv.DictReader(open(SRC)):
    M,N,K=int(r["M"]),int(r["N"]),int(r["K"])
    o=num(r,"o_us"); d=num(r,"default_us"); osl=num(r,"o_slow"); dsl=num(r,"default_slow")
    if d is None or d<=0:
        continue
    ok = (o is not None) and (o != INF) and (o > 0)   # Origami pick runnable?
    rows.append(dict(M=M,N=N,K=K,b=bof(M),nk=f"{N}x{K}",
                     ok=ok, speed=(d/o if ok else None), oslow=osl, dslow=dsl))

def agg(items):
    run=[x for x in items if x["ok"]]
    sp=[x["speed"] for x in run]
    wins=sum(1 for x in run if x["speed"]>=0.999)   # fails count as non-wins
    dsl=[x["dslow"] for x in items if x["dslow"] is not None]
    return dict(n=len(items), n_run=len(run),
                fail_rate=round((len(items)-len(run))/len(items),3),
                speed_med=round(st.median(sp),3) if sp else 0.0,
                speed_mean=round(st.mean(sp),3) if sp else 0.0,
                winrate=round(wins/len(items),3),
                oslow_med=round(st.median([x["oslow"] for x in run if x["oslow"] is not None]),3) if run else None,
                dslow_med=round(st.median(dsl),3) if dsl else None)

# per (bucket, nk)
cell=defaultdict(list)
for x in rows: cell[(x["b"],x["nk"])].append(x)
cells=[]
for (b,nk),it in cell.items():
    N,K=nk.split("x")
    cells.append(dict(bucket=b,nk=nk,N=int(N),K=int(K),**agg(it)))

# marginals
by_nk=defaultdict(list); by_b=defaultdict(list)
for x in rows: by_nk[x["nk"]].append(x); by_b[x["b"]].append(x)
nk_marg=[]
for nk,it in by_nk.items():
    N,K=nk.split("x"); nk_marg.append(dict(nk=nk,N=int(N),K=int(K),**agg(it)))
nk_marg.sort(key=lambda r:(r["N"]*r["K"],r["N"]))
b_marg=[dict(bucket=b,**agg(it)) for b,it in by_b.items()]
border={v:i for i,v in enumerate(BLABEL.values())}
b_marg.sort(key=lambda r:border[r["bucket"]])

overall=agg(rows)
meta=dict(total=len(rows),
          buckets=[b["bucket"] for b in b_marg],
          nks=[m["nk"] for m in nk_marg],
          overall=overall)
print(json.dumps(dict(meta=meta, cells=cells, nk_marg=nk_marg, b_marg=b_marg)))
