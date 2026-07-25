import torch
import aiter
from aiter import dtypes

M, N, K = 64, 2048, 3072
bn, bk = 128, 128
scale_n = (N + bn - 1) // bn
scale_k = (K + bk - 1) // bk
x = (torch.rand((M, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
w = (torch.rand((N, K), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
xs = torch.rand([M, scale_k], dtype=dtypes.fp32, device="cuda")
ws = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
out = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")

print("triggering build of module_gemm_a8w8_blockscale_tune ...", flush=True)
o = aiter.gemm_a8w8_blockscale_tune(x, w, xs, ws, out, 0, 0)
torch.cuda.synchronize()
print("BUILD_OK tune module ready; sample out:", tuple(o.shape), o.dtype, flush=True)
