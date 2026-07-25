#!/usr/bin/env python3
"""Parse the batched Omniperf pmc_perf.csv: isolate the fp8 GEMM dispatches,
map them (in Dispatch order, R per job) back to each (shape,kernelId) job,
compute arithmetic intensity + achieved TFLOPS, and join to variant labels.
Ceilings come from Omniperf's empirical roofline.csv."""
import csv, json
import pandas as pd

R = 3
WL = "/home/demantri/origami_bench/experiments/roofline/roof_batch"
jobs = json.load(open("/home/demantri/origami_bench/experiments/roofline/roof_jobs.json"))
JOB = jobs["jobs"]; SAMP = jobs["sampled"]

# --- ceilings (empirical, from Omniperf) ---
rc = next(csv.DictReader(open(f"{WL}/roofline.csv")))
ceil = dict(fp8_gflops=float(rc["MFMAF8Flops"]), hbm_gbs=float(rc["HBMBw"]),
            fp16_gflops=float(rc["MFMAF16Flops"]), bf16_gflops=float(rc["MFMABF16Flops"]),
            l2_gbs=float(rc["L2Bw"]))

df = pd.read_csv(f"{WL}/pmc_perf.csv")
g = df[df["SQ_INSTS_VALU_MFMA_MOPS_F8"].fillna(0) > 0].copy()
g = g.sort_values("Dispatch_ID").reset_index(drop=True)
n = len(g)
print(f"gemm dispatches: {n}  (expected {len(JOB)*R})")
assert n == len(JOB) * R, f"dispatch/job mismatch: {n} != {len(JOB)*R}"

def hbm_bytes(row):
    rd = row["TCC_EA0_RDREQ_sum"]; rd32 = row["TCC_EA0_RDREQ_32B_sum"]
    bub = row.get("TCC_BUBBLE_sum", 0) or 0
    wr = row["TCC_EA0_WRREQ_sum"]; wr64 = row["TCC_EA0_WRREQ_64B_sum"]
    fetch = bub * 128 + (rd - bub - rd32) * 64 + rd32 * 32
    write = (wr - wr64) * 32 + wr64 * 64
    return fetch + write

job_metrics = {}   # (dataset,M,N,K,kid) -> (ai, tflops, hbm_gbs_ach, dur_us)
for i, j in enumerate(JOB):
    chunk = g.iloc[i*R:(i+1)*R]
    M, N, K = j["M"], j["N"], j["K"]
    flops = 2.0 * M * N * K
    durs = (chunk["End_Timestamp"] - chunk["Start_Timestamp"]).astype(float)  # ns
    byts = chunk.apply(hbm_bytes, axis=1).astype(float)
    dur_ns = float(durs.median()); b = float(byts.median())
    tflops = flops / (dur_ns * 1e-9) / 1e12
    ai = flops / b if b > 0 else 0.0
    ach_bw = b / (dur_ns * 1e-9) / 1e9  # achieved HBM GB/s
    job_metrics[(j["dataset"], M, N, K, j["kid"])] = dict(
        ai=round(ai, 3), tflops=round(tflops, 1), hbm_gbs=round(ach_bw, 1), dur_us=round(dur_ns/1e3, 2))

# --- build points per sampled shape x variant ---
points = []
for s in SAMP:
    for var, kid in (("origami", s["origami_kid"]), ("oracle", s["oracle_kid"]), ("default", s["default_kid"])):
        m = job_metrics.get((s["dataset"], s["M"], s["N"], s["K"], kid))
        if not m: continue
        points.append(dict(dataset=s["dataset"], M=s["M"], N=s["N"], K=s["K"], bucket=s["bucket"],
                           variant=var, kid=kid, ai=m["ai"], tflops=m["tflops"],
                           hbm_gbs=m["hbm_gbs"], dur_us=m["dur_us"]))

out = dict(ceilings=ceil, points=points, n_shapes=len(SAMP), n_jobs=len(JOB))
json.dump(out, open("/home/demantri/origami_bench/experiments/roofline/roof_points.json", "w"), indent=1)
print(f"ceilings: fp8={ceil['fp8_gflops']/1e6:.2f} PFLOPS, HBM={ceil['hbm_gbs']/1e3:.2f} TB/s")
print(f"points: {len(points)}")
# quick sanity: show a few
for p in points[:8]:
    print(f"  [{p['dataset']:8}] M={p['M']:6} {p['variant']:7} kid={p['kid']:2} "
          f"AI={p['ai']:8.2f} F/B  {p['tflops']:7.1f} TFLOPS  {p['hbm_gbs']:7.0f} GB/s  {p['dur_us']}us")
