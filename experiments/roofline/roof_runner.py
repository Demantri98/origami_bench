"""Run ONE CK a8w8_blockscale kernel (by kernelId) on one (M,N,K) shape, many
times, so rocprof-compute can profile it for a roofline point. splitK=0."""
import sys
import torch
import aiter
from aiter import dtypes

M, N, K, kid = (int(x) for x in sys.argv[1:5])
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 40
bn, bk = 128, 128
sk = (K + bk - 1) // bk
sn = (N + bn - 1) // bn
x = (torch.rand((M, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
w = (torch.rand((N, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
xs = torch.rand([M, sk], dtype=dtypes.fp32, device="cuda")
ws = torch.rand([sn, sk], dtype=dtypes.fp32, device="cuda")
o = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
for _ in range(3):
    aiter.gemm_a8w8_blockscale_tune(x, w, xs, ws, o, kid, 0)
torch.cuda.synchronize()
for _ in range(iters):
    aiter.gemm_a8w8_blockscale_tune(x, w, xs, ws, o, kid, 0)
torch.cuda.synchronize()
