#!/usr/bin/env python3
"""Small direct latency probe for a specific aiter CK candidate kernelId."""

import argparse
import statistics

import torch
import aiter
from aiter import dtypes


parser = argparse.ArgumentParser()
parser.add_argument("--shape", required=True)
parser.add_argument("--kid", type=int, required=True)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--iters", type=int, default=100)
args = parser.parse_args()
dims = dict(part.split("=") for part in args.shape.split(","))
M, N, K = (int(dims[key]) for key in ("M", "N", "K"))
x = (torch.rand((M, K), dtype=dtypes.fp16, device="cuda") * 0.1).to(dtypes.fp8)
w = (torch.rand((N, K), dtype=dtypes.fp16, device="cuda") * 0.1).to(dtypes.fp8)
xs = torch.rand((M, K // 128), dtype=dtypes.fp32, device="cuda")
ws = torch.rand(((N + 127) // 128, K // 128), dtype=dtypes.fp32, device="cuda")
out = torch.empty((M, N), dtype=dtypes.bf16, device="cuda")
for _ in range(args.warmup):
    aiter.gemm_a8w8_blockscale_tune(x, w, xs, ws, out, args.kid, 0)
torch.cuda.synchronize()
times = []
for _ in range(args.iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    aiter.gemm_a8w8_blockscale_tune(x, w, xs, ws, out, args.kid, 0)
    end.record()
    end.synchronize()
    times.append(float(start.elapsed_time(end)))
print(f"median_ms: {statistics.median(times):.8f}")
print(f"min_ms: {min(times):.8f}")
print(f"max_ms: {max(times):.8f}")
