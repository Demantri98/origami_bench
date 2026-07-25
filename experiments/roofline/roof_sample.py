#!/usr/bin/env python3
"""Sample 2 shapes/M-bucket from each dataset and resolve the 3 CK kernel
variants (Origami pick / oracle=exhaustive-best / default kid 7) per shape.
Emits roof_jobs.json: sampled shapes + an ordered dedup'd (shape,kernelId) job
list for one batched Omniperf roofline profiling pass."""
import csv, json, math
from collections import defaultdict

DATASETS = {
    "tuned": "/home/demantri/origami_bench/results/estimation_vs_measured_tuned.csv",
    "novel_nk": "/home/demantri/origami_bench/results/estimation_vs_measured_novel_nk.csv",
}
BUCKETS = [(0,16),(16,64),(64,256),(256,1024),(1024,4096),(4096,10**12)]
BLABEL = {b: l for b,l in zip(BUCKETS,["<=16","16-64","64-256","256-1024","1024-4096",">4096"])}
DEFAULT_KID = 7

def bof(m):
    for b in BUCKETS:
        if b[0] < m <= b[1]: return b

def fnum(r,k):
    try: return float(r[k])
    except: return math.inf

sampled = []
for ds, path in DATASETS.items():
    rows = list(csv.DictReader(open(path)))
    by_b = defaultdict(list)
    for r in rows:
        M = int(r["M"])
        # need runnable Origami + oracle + default
        if fnum(r,"o_us")==math.inf or fnum(r,"oracle_us")==math.inf or fnum(r,"default_us")==math.inf:
            continue
        by_b[bof(M)].append(r)
    for b in BUCKETS:
        pool = sorted(by_b.get(b, []), key=lambda r: int(r["M"]))
        if not pool: continue
        diff = [r for r in pool if int(r["o_kid"]) != int(r["oracle_kid"])]
        same = [r for r in pool if int(r["o_kid"]) == int(r["oracle_kid"])]
        picks = []
        if diff: picks.append(diff[len(diff)//2])          # an Origami "miss"
        if same: picks.append(same[len(same)//2])          # an Origami "hit"
        for r in pool:                                       # fill to 2, spread
            if len(picks) >= 2: break
            if r not in picks: picks.append(r)
        for r in picks[:2]:
            sampled.append(dict(dataset=ds, M=int(r["M"]), N=int(r["N"]), K=int(r["K"]),
                                 origami_kid=int(r["o_kid"]), oracle_kid=int(r["oracle_kid"]),
                                 default_kid=DEFAULT_KID, bucket=BLABEL[b]))

# ordered dedup'd jobs: per sampled shape, the distinct kernelIds to profile
jobs = []
for s in sampled:
    seen = {}
    for var, kid in (("origami", s["origami_kid"]), ("oracle", s["oracle_kid"]), ("default", s["default_kid"])):
        seen.setdefault(kid, []).append(var)
    for kid, vs in seen.items():
        jobs.append(dict(idx=len(jobs), dataset=s["dataset"], M=s["M"], N=s["N"], K=s["K"],
                         kid=kid, variants=vs, bucket=s["bucket"]))

out = dict(sampled=sampled, jobs=jobs)
json.dump(out, open("/home/demantri/origami_bench/experiments/roofline/roof_jobs.json","w"), indent=1)
print(f"sampled shapes: {len(sampled)} ({sum(1 for s in sampled if s['dataset']=='tuned')} tuned, "
      f"{sum(1 for s in sampled if s['dataset']=='novel_nk')} novel_nk)")
print(f"profiling jobs (unique shape,kid): {len(jobs)}")
for s in sampled:
    print(f"  [{s['dataset']:8}] M={s['M']:6} N={s['N']:6} K={s['K']:6} bucket={s['bucket']:10} "
          f"origami={s['origami_kid']} oracle={s['oracle_kid']} default={s['default_kid']}")
