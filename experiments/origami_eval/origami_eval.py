#!/usr/bin/env python3
"""Evaluate Origami's ESTIMATION model against MEASURED ground truth on the
shapes in aiter's a8w8_blockscale_tuned_gemm.csv.

For each sampled (M,N,K):
  * Origami estimation ranks the 19 CK candidate kernelIds -> pick (splitK=0).
  * We MEASURE all 19 kernelIds on the GPU (tune op, splitK=0) -> true oracle.
  * We also measure the CSV's tuned kernelId and the default/backup kernel.
Compares Origami pick vs measured oracle (NOT vs the CSV's claimed best).
"""
import argparse, csv, os, sys, random, time, statistics as st
from collections import defaultdict

FIELDS = ["M","N","K","o_kid","hint_mode","o_us","o_rank","o_slow","oracle_kid",
          "oracle_us","csv_kid","csv_us","csv_slow","default_us","default_slow","n_ok"]

import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest

sys.path.insert(0, "/home/demantri/origami_bench")
sys.path.insert(0, "/home/demantri/origami_bench/tools")
import origami
import ck_kernel_map
from common.families import get_family

CSV = "/home/demantri/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv"
BLOCK = (128, 128)
DEFAULT_KID = 7  # a8w8_blockscale ...256x16x128x256..._v1 = C++ default/backup

def gen_data(M, N, K, seed=0):
    torch.manual_seed(seed)
    bn, bk = BLOCK
    sk = (K + bk - 1)//bk; sn = (N + bn - 1)//bn
    x = (torch.rand((M,K), dtype=dtypes.fp16, device="cuda")/10).to(dtypes.fp8)
    w = (torch.rand((N,K), dtype=dtypes.fp16, device="cuda")/10).to(dtypes.fp8)
    xs = torch.rand([M, sk], dtype=dtypes.fp32, device="cuda")
    ws = torch.rand([sn, sk], dtype=dtypes.fp32, device="cuda")
    out = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
    return x, w, xs, ws, out

def measure(kid, x, w, xs, ws, out, warmup, iters):
    try:
        _, us = run_perftest(aiter.gemm_a8w8_blockscale_tune, x, w, xs, ws, out, kid, 0,
                             num_warmup=warmup, num_iters=iters,
                             num_rotate_args=1, use_cuda_event=True)
        torch.cuda.synchronize()
        return float(us)
    except Exception:
        return float("inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=60)
    ap.add_argument("--all", action="store_true", help="run EVERY unique ck shape in the CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap total shapes")
    ap.add_argument("--out", default="/home/demantri/origami_bench/results/estimation_vs_measured.csv")
    args = ap.parse_args()

    # --- load + dedupe CSV shapes (capture tuned kid) ---
    rows = {}
    for r in csv.DictReader(open(CSV)):
        if r["libtype"] != "ck":
            continue
        key = (int(r["M"]), int(r["N"]), int(r["K"]))
        rows[key] = int(r["kernelId"])  # csv tuned kid

    # --- stratified sample across M-buckets ---
    buckets = [(0,16),(16,64),(64,256),(256,1024),(1024,4096),(4096,10**9)]
    by_bucket = defaultdict(list)
    for key in rows:
        m = key[0]
        for lo,hi in buckets:
            if lo < m <= hi:
                by_bucket[(lo,hi)].append(key); break
    if args.all:
        sample = sorted(rows.keys())
    else:
        rng = random.Random(args.seed)
        sample = []
        for b in buckets:
            pool = sorted(by_bucket[b])
            rng.shuffle(pool)
            sample += pool[:args.per_bucket]
        sample = sorted(set(sample))
    if args.limit:
        sample = sample[:args.limit]

    # --- Origami estimation setup ---
    fam = get_family("a8w8_blockscale")
    hw = origami.get_hardware_for_device(0)
    n_cu = int(hw.N_CU)
    base_cfgs = ck_kernel_map.build_configs(origami, hw, fam, "", occupancy=2)
    # The estimation model's short-circuit REQUIRES a non-temporal hint for
    # memory-bound skinny shapes (else every candidate is marked infeasible) and
    # REJECTS it for large M. A correct integration sets it conditionally, so we
    # rank with a hint fallback: base (temporal) -> non-temporal B -> non-temporal A.
    ntb_cfgs = ck_kernel_map.build_configs(origami, hw, fam, "", occupancy=2)
    for _, c in ntb_cfgs: c.cache_hints_b = 4
    nta_cfgs = ck_kernel_map.build_configs(origami, hw, fam, "", occupancy=2)
    for _, c in nta_cfgs: c.cache_hints_a = 4

    def rank_with_fallback(prob):
        for mode, cfgs in (("base", base_cfgs), ("nt_b", ntb_cfgs), ("nt_a", nta_cfgs)):
            scored = ck_kernel_map.rank_kernelids(origami, hw, prob, cfgs)
            if scored:
                return scored, mode
        return [], "none"

    print(f"Eval: {len(sample)} shapes ({'ALL' if args.all else 'stratified'}), device gfx950 "
          f"N_CU={n_cu}, measure all 19 kids @ splitK=0 (warmup={args.warmup} iters={args.iters})\n",
          flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_fh = open(args.out, "w", newline="")
    out_wr = csv.DictWriter(out_fh, fieldnames=FIELDS)
    out_wr.writeheader(); out_fh.flush()

    per_shape = []
    t0 = time.time()
    for i,(M,N,K) in enumerate(sample):
        prob = origami.problem_t()
        prob.size = origami.dim3_t(M,N,K); prob.batch=1
        prob.a_transpose=origami.transpose_t.T; prob.b_transpose=origami.transpose_t.N
        f8=origami.string_to_datatype("f8"); bf16=origami.string_to_datatype("bf16")
        prob.a_dtype=f8; prob.b_dtype=f8; prob.c_dtype=bf16; prob.d_dtype=bf16; prob.mi_dtype=f8
        scored, hint_mode = rank_with_fallback(prob)
        o_kid = scored[0][0] if scored else -1

        x,w,xs,ws,out = gen_data(M,N,K,args.seed)
        meas = {}
        for kid,_ in base_cfgs:
            us = measure(kid, x,w,xs,ws,out, args.warmup, args.iters)
            if us != float("inf"):
                meas[kid] = us
        del x,w,xs,ws,out; torch.cuda.empty_cache()
        if not meas:
            continue
        oracle_kid = min(meas, key=meas.get); oracle_us = meas[oracle_kid]
        ranked = sorted(meas, key=meas.get)
        o_us = meas.get(o_kid, float("inf"))
        o_rank = ranked.index(o_kid)+1 if o_kid in meas else -1
        o_slow = o_us/oracle_us if o_us!=float("inf") else float("inf")
        csv_kid = rows[(M,N,K)]
        csv_us = meas.get(csv_kid, float("inf")); csv_slow = csv_us/oracle_us if csv_us!=float("inf") else float("inf")
        d_us = meas.get(DEFAULT_KID, float("inf")); d_slow = d_us/oracle_us if d_us!=float("inf") else float("inf")
        per_shape.append(dict(M=M,N=N,K=K, o_kid=o_kid, hint_mode=hint_mode,
                              o_us=round(o_us,3), o_rank=o_rank,
                              o_slow=round(o_slow,4), oracle_kid=oracle_kid, oracle_us=round(oracle_us,3),
                              csv_kid=csv_kid, csv_us=round(csv_us,3), csv_slow=round(csv_slow,4),
                              default_us=round(d_us,3), default_slow=round(d_slow,4),
                              n_ok=len(meas)))
        out_wr.writerow(per_shape[-1]); out_fh.flush()
        if (i+1)%100==0:
            el=time.time()-t0; rate=(i+1)/el; eta=(len(sample)-(i+1))/rate
            print(f"  ...{i+1}/{len(sample)} done  ({el:.0f}s elapsed, {rate:.1f} shapes/s, ETA {eta/60:.1f} min)", flush=True)

    out_fh.close()

    # --- aggregate ---
    def stats(vals):
        vals=[v for v in vals if v!=float("inf")]
        if not vals: return None
        s=sorted(vals)
        return dict(mean=st.mean(vals), median=st.median(vals),
                    p90=s[min(len(s)-1,int(0.9*len(s)))], mx=max(vals), n=len(vals))
    def fmt(d):
        return "n/a" if d is None else f"mean={d['mean']:.3f}x median={d['median']:.3f}x p90={d['p90']:.3f}x max={d['mx']:.3f}x"
    def bucket_of(m):
        for lo,hi in buckets:
            if lo<m<=hi: return (lo,hi)

    print("\n"+"#"*100)
    print(f"RESULTS: Origami ESTIMATION vs MEASURED oracle  ({len(per_shape)} shapes, splitK=0)")
    print("  (coverage = shapes where the model produced a feasible pick; "
          "slowdown = picked_us / measured_best_us; hit@1 among COVERED)")
    for b in buckets:
        rs=[r for r in per_shape if bucket_of(r["M"])==b]
        if not rs: continue
        cov=[r for r in rs if r["o_kid"]!=-1 and r["o_slow"]!=float("inf")]
        coverage=len(cov)/len(rs)*100
        hit=(sum(1 for r in cov if r["o_kid"]==r["oracle_kid"])/len(cov)*100) if cov else 0
        hints=defaultdict(int)
        for r in cov: hints[r["hint_mode"]]+=1
        print(f"\n  M in ({b[0]},{b[1]}]  ({len(rs)} shapes)  coverage={coverage:.0f}%  hint_modes={dict(hints)}")
        print(f"    Origami : hit@1={hit:5.1f}%  slowdown {fmt(stats([r['o_slow'] for r in cov]))}")
        print(f"    CSV pick:              slowdown {fmt(stats([r['csv_slow'] for r in rs]))}  (measured @sk0)")
        print(f"    default7:              slowdown {fmt(stats([r['default_slow'] for r in rs]))}")

    rs=per_shape
    cov=[r for r in rs if r["o_kid"]!=-1 and r["o_slow"]!=float("inf")]
    coverage=len(cov)/len(rs)*100
    hit=(sum(1 for r in cov if r["o_kid"]==r["oracle_kid"])/len(cov)*100) if cov else 0
    within5=(sum(1 for r in cov if r["o_slow"]<=1.05)/len(cov)*100) if cov else 0
    within10=(sum(1 for r in cov if r["o_slow"]<=1.10)/len(cov)*100) if cov else 0
    print("\n  "+"="*84)
    print(f"  OVERALL ({len(rs)} shapes equal-weight across buckets):")
    print(f"    Coverage (feasible pick) = {coverage:.1f}%")
    print(f"    Among covered: hit@1={hit:.1f}%  within-1.05x={within5:.1f}%  within-1.10x={within10:.1f}%")
    print(f"    Origami  slowdown vs oracle: {fmt(stats([r['o_slow'] for r in cov]))}")
    print(f"    CSV pick slowdown vs oracle: {fmt(stats([r['csv_slow'] for r in rs]))}")
    print(f"    default7 slowdown vs oracle: {fmt(stats([r['default_slow'] for r in rs]))}")
    print(f"    Saved per-shape -> {args.out}")

if __name__=="__main__":
    main()
