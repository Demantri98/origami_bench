import csv, os
import torch
import aiter
from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx

TUNED = "/home/demantri/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv"
SYNTH = "/home/demantri/origami_bench/datasets/a8w8_blockscale_gfx950_synth.csv"
block_shape = (128, 128)

# Tuned (N,K) combos present in the CSV.
tuned_nk = set()
with open(TUNED) as f:
    for r in csv.DictReader(f):
        tuned_nk.add((int(r["N"]), int(r["K"])))
print("tuned (N,K) combos:", sorted(tuned_nk))

# Take the M distribution from the synth file, but remap onto NEW (N,K) combos
# that are NOT in the tuned CSV (all divisible by 128).
Ms = sorted({int(r["M"]) for r in csv.DictReader(open(SYNTH))})
new_nk = [(2048, 4096), (5120, 3072), (1024, 2048), (4096, 4096), (7168, 8192)]
for nk in new_nk:
    assert nk not in tuned_nk, f"{nk} unexpectedly in tuned set"

print(f"\ngfx={get_gfx()} cu_num={get_cu_num()}")

# ---- Analytical: dispatch over ALL remapped rows ----
total = 0; fb = 0
for (N, K) in new_nk:
    for M in Ms:
        total += 1
        if get_CKGEMM_config(M, N, K, TUNED) is None:
            fb += 1
print(f"\nDISPATCH over {total} remapped rows (synth M x new N,K):")
print(f"  fallback (backup CK) : {fb}")
print(f"  tuned hits           : {total - fb}")

# ---- Actually invoke a sample and confirm the backup kernel runs ----
def make_inputs(m, n, k):
    bn, bk = block_shape
    sk = (k + bk - 1) // bk; sn = (n + bn - 1) // bn
    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    w = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    xs = torch.rand([m, sk], dtype=dtypes.fp32, device="cuda")
    ws = torch.rand([sn, sk], dtype=dtypes.fp32, device="cuda")
    return x, w, xs, ws

sample_M = [Ms[0], Ms[len(Ms)//2], Ms[-1]]
print(f"\n{'M':>7} {'N':>6} {'K':>6}  {'result':>10}  kernel")
for (N, K) in new_nk:
    for M in sample_M:
        cfg = get_CKGEMM_config(M, N, K, TUNED)
        x, w, xs, ws = make_inputs(M, N, K)
        out = aiter.gemm_a8w8_blockscale(x, w, xs, ws, dtypes.bf16)
        torch.cuda.synchronize()
        ok = tuple(out.shape) == (M, N) and out.dtype == dtypes.bf16
        label = "FALLBACK" if cfg is None else "TUNED:" + cfg["libtype"]
        kname = "(default/backup CK kernel)" if cfg is None else cfg["kernelName"]
        assert ok
        print(f"{M:>7} {N:>6} {K:>6}  {label:>10}  {kname}")
