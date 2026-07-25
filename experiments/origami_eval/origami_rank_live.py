#!/usr/bin/env python3
"""Rank the a8w8_blockscale CK candidate templates with LIVE Origami and join
against the measured sweep, for the exact shapes we benchmarked earlier."""
import csv, os, sys
from collections import defaultdict

sys.path.insert(0, "/home/demantri/origami_bench")
sys.path.insert(0, "/home/demantri/origami_bench/tools")
import origami
import ck_kernel_map
from common.families import get_family

MEAS = "/home/demantri/origami_bench/results/ck_template_baseline.csv"
SHAPES = [(64,2048,3072),(256,2048,3072),(2048,2048,3072),(8192,2048,3072),(64,3072,6144)]

fam = get_family("a8w8_blockscale")
# device hardware (matches the box we measured on: gfx950, 256 CU)
hw = origami.get_hardware_for_device(0)
n_cu = int(getattr(hw, "N_CU", 0) or 0)
od = fam.origami_dtypes[""]
ktable = ck_kernel_map.load_kernel_table(fam)
configs = ck_kernel_map.build_configs(origami, hw, fam, "", occupancy=2)

def make_problem(m, n, k):
    p = origami.problem_t()
    p.size = origami.dim3_t(m, n, k)
    p.batch = 1
    p.a_transpose = origami.transpose_t.T
    p.b_transpose = origami.transpose_t.N
    def dt(tok, fb="f8"):
        for t in (tok, fb):
            try: return origami.string_to_datatype(t)
            except Exception: pass
        return origami.string_to_datatype("f16")
    p.a_dtype = dt(od["a"]); p.b_dtype = dt(od["b"])
    p.d_dtype = dt(od["out"], "bf16"); p.c_dtype = p.d_dtype
    p.mi_dtype = dt(od["mi"], od["a"])
    p.a_mx_block_size = od.get("mx", 0); p.b_mx_block_size = od.get("mx", 0)
    return p

# measured[(M,N,K)][kid] = us
measured = defaultdict(dict); mtile = {}
for r in csv.DictReader(open(MEAS)):
    if r["status"] != "ok": continue
    measured[(int(r["M"]),int(r["N"]),int(r["K"]))][int(r["kernelId"])] = float(r["us"])
    mtile[int(r["kernelId"])] = r["tile_MxNxK"]

print(f"LIVE Origami ranking  (device gfx950, N_CU={n_cu}, occupancy=2)\n")
for (M,N,K) in SHAPES:
    prob = make_problem(M,N,K)
    scored = ck_kernel_map.rank_kernelids(origami, hw, prob, configs)  # [(kid,cyc)] asc
    pred_rank = {kid:i for i,(kid,_) in enumerate(scored, start=1)}
    pred_cyc = dict(scored)
    meas = measured.get((M,N,K), {})
    best_kid = min(meas, key=meas.get); best_us = meas[best_kid]
    meas_rank = {kid:i for i,(kid,_) in enumerate(sorted(meas.items(),key=lambda kv:kv[1]),1)}
    o_kid, o_cyc = scored[0]
    o_us = meas.get(o_kid)
    slow = o_us/best_us if o_us else float("nan")
    print("="*96)
    print(f"SHAPE (M,N,K)=({M},{N},{K})")
    print(f"  Origami #1: kid {o_kid:2d} [{mtile.get(o_kid,'?'):>11}] {o_cyc:.0f} cyc "
          f"-> measured {o_us:.3f} us, meas-rank {meas_rank.get(o_kid,'?')}/{len(meas)}, {slow:.2f}x vs best")
    print(f"  Measured best: kid {best_kid:2d} [{mtile.get(best_kid,'?'):>11}] {best_us:.3f} us")
    print(f"  {'kid  tile':<20} pred_cyc  pred_rk | meas_us  meas_rk")
    for kid,_ in sorted(meas.items(), key=lambda kv: kv[1])[:8]:
        pc = pred_cyc.get(kid); pr = pred_rank.get(kid,'-')
        print(f"  kid {kid:2d} [{mtile.get(kid,'?'):>11}] {('%.0f'%pc) if pc else 'INF':>8}  "
              f"{str(pr):>6} | {meas[kid]:7.3f}  {meas_rank[kid]:>6}")
    print()
