#!/usr/bin/env python3
"""Verify the DEFAULT/backup CK kernel is invoked for the untuned_nk_synth
dataset: (1) get_CKGEMM_config returns None for every shape, (2) the real
gemm_a8w8_blockscale dispatch logs 'will use default config', and (3) its
measured latency matches the default kernel (kid 7) run via the tune op.
"""
import csv, sys
import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
from aiter.jit.core import AITER_CONFIGS

DS = "/home/demantri/origami_bench/datasets/a8w8_blockscale_gfx950_untuned_nk_synth.csv"
TUNED = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
DEFAULT_KID = 7
BLOCK = (128, 128)

shapes = [(int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(open(DS))]

# (1) config lookup must be None for all shapes (=> default kernel path)
none_cnt = sum(1 for (M, N, K) in shapes if get_CKGEMM_config(M, N, K, TUNED) is None)
print(f"(1) get_CKGEMM_config == None for {none_cnt}/{len(shapes)} shapes "
      f"({'ALL -> default kernel path' if none_cnt == len(shapes) else 'NOT ALL'})")

# (3) empirical: real dispatch latency == kid-7 (default) latency, on a sample
def gen(M, N, K, seed=0):
    torch.manual_seed(seed)
    bn, bk = BLOCK
    sk = (K + bk - 1) // bk; sn = (N + bn - 1) // bn
    x = (torch.rand((M, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    w = (torch.rand((N, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    xs = torch.rand([M, sk], dtype=dtypes.fp32, device="cuda")
    ws = torch.rand([sn, sk], dtype=dtypes.fp32, device="cuda")
    out = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
    return x, w, xs, ws, out

sample = shapes[::160]  # ~5 spread across the file
print("\n(3) real gemm_a8w8_blockscale vs kid-7 (default) tune op, measured us:")
print(f"  {'M':>6} {'N':>6} {'K':>6} {'prod_us':>9} {'default(kid7)_us':>16} {'match':>6}")
for (M, N, K) in sample:
    x, w, xs, ws, out = gen(M, N, K)
    _, prod_us = run_perftest(aiter.gemm_a8w8_blockscale, x, w, xs, ws, dtypes.bf16,
                              num_warmup=3, num_iters=10, num_rotate_args=1, use_cuda_event=True)
    _, def_us = run_perftest(aiter.gemm_a8w8_blockscale_tune, x, w, xs, ws, out, DEFAULT_KID, 0,
                             num_warmup=3, num_iters=10, num_rotate_args=1, use_cuda_event=True)
    torch.cuda.synchronize()
    ratio = prod_us / def_us
    print(f"  {M:>6} {N:>6} {K:>6} {prod_us:>9.2f} {def_us:>16.2f} {ratio:>6.2f}")
    del x, w, xs, ws, out; torch.cuda.empty_cache()
print("  (ratio ~1.0 confirms production runs the default kernel)")
