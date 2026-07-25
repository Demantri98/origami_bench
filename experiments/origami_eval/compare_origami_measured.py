#!/usr/bin/env python3
"""Join Origami's predicted ranking (cached) vs measured per-template latency."""
import csv, json
from collections import defaultdict

ORIG = "/home/demantri/origami_bench/results/a8w8_blockscale_origami.csv"
MEAS = "/home/demantri/origami_bench/results/ck_measured_for_origami_shapes.csv"

# measured[(M,N,K)][kid] = (us, tile)
measured = defaultdict(dict)
mtile = {}
for r in csv.DictReader(open(MEAS)):
    if r["status"] != "ok":
        continue
    key = (int(r["M"]), int(r["N"]), int(r["K"]))
    kid = int(r["kernelId"])
    measured[key][kid] = float(r["us"])
    mtile[kid] = r["tile_MxNxK"]

# origami predicted per shape: full topk list
orig = {}
for r in csv.DictReader(open(ORIG)):
    if r.get("available") != "True" or not r.get("topk_json"):
        continue
    key = (int(r["M"]), int(r["N"]), int(r["K"]))
    if key not in measured:
        continue
    tk = json.loads(r["topk_json"])
    orig[key] = {"best_kid": int(r["best_kernelId"]), "topk": tk}

for key in sorted(orig):
    M, N, K = key
    meas = measured[key]
    # measured ranking (ascending us)
    meas_rank = {kid: i for i, (kid, _) in
                 enumerate(sorted(meas.items(), key=lambda kv: kv[1]), start=1)}
    best_kid = min(meas, key=meas.get)
    best_us = meas[best_kid]

    o_best = orig[key]["best_kid"]
    o_best_us = meas.get(o_best)
    slow = (o_best_us / best_us) if o_best_us else None

    print("=" * 92)
    print(f"SHAPE (M,N,K)=({M},{N},{K})")
    print(f"  Origami #1 pick : kid {o_best:2d} [{mtile.get(o_best,'?'):>11}]  "
          f"-> measured {o_best_us:.3f} us, measured-rank {meas_rank.get(o_best,'?')}/{len(meas)}"
          + (f", {slow:.2f}x vs best" if slow else ""))
    print(f"  Measured BEST   : kid {best_kid:2d} [{mtile.get(best_kid,'?'):>11}]  "
          f"-> {best_us:.3f} us")
    print(f"  {'Origami top-5 (pred cycles)':<46} | measured us | meas-rank")
    print(f"  {'-'*46}-+-------------+----------")
    for e in orig[key]["topk"]:
        kid = e["kernelId"]; cyc = e["latency_cycles"]
        mus = meas.get(kid)
        tie = ""
        # flag collisions: same predicted cycles as another entry
        same = [x["kernelId"] for x in orig[key]["topk"]
                if abs(x["latency_cycles"] - cyc) < 1e-6 and x["kernelId"] != kid]
        if same:
            tie = f"  <=tie with {same}"
        mus_s = f"{mus:.3f}" if mus is not None else "n/a"
        rk = meas_rank.get(kid, "?")
        print(f"  kid {kid:2d} [{mtile.get(kid,'?'):>11}]  {cyc:10.0f} cyc"
              f"           | {mus_s:>9}   | {str(rk):>3}{tie}")
    print()

# summary
print("#" * 92)
print("SUMMARY: Origami pick quality (measured)")
for key in sorted(orig):
    meas = measured[key]
    best_us = min(meas.values())
    o_best = orig[key]["best_kid"]
    o_us = meas.get(o_best)
    slow = o_us / best_us if o_us else float("nan")
    rank = sorted(meas, key=meas.get).index(o_best) + 1 if o_best in meas else -1
    print(f"  M={key[0]:5d} N={key[1]} K={key[2]}: origami kid {o_best:2d} "
          f"is measured-rank {rank}/{len(meas)}, {slow:.2f}x vs best")
