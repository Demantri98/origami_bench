import csv, os
import torch
import aiter
from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx

TUNED = "/home/demantri/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv"
SYNTH = "/home/demantri/origami_bench/datasets/a8w8_blockscale_gfx950_synth.csv"
block_shape = (128, 128)

# Load all synth shapes, grouped by (N,K)
rows = []
with open(SYNTH) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["M"]), int(r["N"]), int(r["K"])))

# Build a representative sample: for each (N,K) combo, take min/median/max M plus a couple mid values.
from collections import defaultdict
by_nk = defaultdict(list)
for (M, N, K) in rows:
    by_nk[(N, K)].append(M)

sample = []
for (N, K), ms in sorted(by_nk.items()):
    ms = sorted(ms)
    picks = {ms[0], ms[len(ms)//4], ms[len(ms)//2], ms[3*len(ms)//4], ms[-1]}
    for m in sorted(picks):
        sample.append((m, N, K))

print(f"gfx={get_gfx()} cu_num={get_cu_num()}")
print(f"Running {len(sample)} representative shapes (of {len(rows)} total)\n")

def make_inputs(m, n, k):
    bn, bk = block_shape
    scale_k = (k + bk - 1) // bk
    scale_n = (n + bn - 1) // bn
    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    w = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
    return x, w, x_scale, w_scale

n_tuned = 0
n_fallback = 0
print(f"{'M':>7} {'N':>6} {'K':>6}  {'result':>10}  kernel")
for (m, n, k) in sample:
    cfg = get_CKGEMM_config(m, n, k, TUNED)
    x, w, xs, ws = make_inputs(m, n, k)
    out = aiter.gemm_a8w8_blockscale(x, w, xs, ws, dtypes.bf16)
    torch.cuda.synchronize()
    ok = tuple(out.shape) == (m, n) and out.dtype == dtypes.bf16
    if cfg is None:
        n_fallback += 1
        label = "FALLBACK"
        kname = "(default/backup CK kernel)"
    else:
        n_tuned += 1
        label = "TUNED:" + cfg["libtype"]
        kname = cfg["kernelName"]
    assert ok, f"bad output for {(m,n,k)}: {out.shape} {out.dtype}"
    print(f"{m:>7} {n:>6} {k:>6}  {label:>10}  {kname}")

print(f"\nSUMMARY over sample: tuned={n_tuned}  fallback(backup CK)={n_fallback}")
