"""Batch app for ONE Omniperf --roof-only pass: runs every (shape,kernelId) job
from roof_jobs.json, R times each, in order. The fp8-MFMA GEMM dispatches then
appear in pmc_perf.csv in this exact job order (R per job) for mapping."""
import json
import torch
import aiter
from aiter import dtypes

R = 3
bn, bk = 128, 128
jobs = json.load(open("/home/demantri/origami_bench/experiments/roofline/roof_jobs.json"))["jobs"]
for j in jobs:
    M, N, K, kid = j["M"], j["N"], j["K"], j["kid"]
    sk = (K + bk - 1) // bk
    sn = (N + bn - 1) // bn
    x = (torch.rand((M, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    w = (torch.rand((N, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    xs = torch.rand([M, sk], dtype=dtypes.fp32, device="cuda")
    ws = torch.rand([sn, sk], dtype=dtypes.fp32, device="cuda")
    o = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
    for _ in range(R):
        aiter.gemm_a8w8_blockscale_tune(x, w, xs, ws, o, kid, 0)
    torch.cuda.synchronize()
    del x, w, xs, ws, o
    torch.cuda.empty_cache()
