#!/usr/bin/env python3
"""KernelForge driver for one editable CK a8w8_blockscale template.

The seeded template always occupies candidate kernelId 0.  This driver supports
KernelForge's validation, benchmark, and single-case profiling contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(os.environ.get("KF_AITER_ROOT", Path(__file__).resolve().parent)).resolve()
INSTANCE_SOURCE = (
    ROOT / "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py"
)
SOURCE_HASH = hashlib.sha256(INSTANCE_SOURCE.read_bytes()).hexdigest()
BUILD_STAMP = ROOT / "aiter/jit/.kf_ab_tune_source.sha256"
TUNE_MODULE = ROOT / "aiter/jit/module_gemm_a8w8_blockscale_tune.so"

# Rebuild once per source revision, not once per validation subprocess.
# AITER_REBUILD=2 retains the incremental object tree.  After the first
# successful launch writes BUILD_STAMP, later driver processes reuse the .so.
previous_hash = BUILD_STAMP.read_text().strip() if BUILD_STAMP.exists() else ""
_SOURCE_NEEDS_BUILD = previous_hash != SOURCE_HASH or not TUNE_MODULE.exists()
_BUILD_STAMP_WRITTEN = not _SOURCE_NEEDS_BUILD
os.environ["AITER_REBUILD"] = "2" if _SOURCE_NEEDS_BUILD else "0"
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402


EDITABLE_KERNEL_ID = 0
SPLIT_K = 0
BLOCK_N = 128
BLOCK_K = 128


def parse_shape(raw: str) -> tuple[int, int, int]:
    if not raw or raw == "default":
        raw = os.environ.get("KF_PRIMARY_SHAPE", "")
    values: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        values[key.strip().upper()] = int(value.strip())
    if not {"M", "N", "K"}.issubset(values):
        raise ValueError(f"shape must provide M,N,K, got {raw!r}")
    return values["M"], values["N"], values["K"]


def make_inputs(M: int, N: int, K: int, mode: str):
    if K % BLOCK_K:
        raise ValueError(f"K={K} must be divisible by {BLOCK_K}")
    torch.manual_seed(0)
    scale_k = (K + BLOCK_K - 1) // BLOCK_K
    scale_n = (N + BLOCK_N - 1) // BLOCK_N

    # Keep values small enough to avoid reference overflow.  Stability mode uses
    # a wider but still finite scale range.
    amplitude = 0.125 if mode == "stability" else 0.1
    x = (torch.rand((M, K), dtype=dtypes.fp16, device="cuda") * amplitude).to(
        dtypes.fp8
    )
    w = (torch.rand((N, K), dtype=dtypes.fp16, device="cuda") * amplitude).to(
        dtypes.fp8
    )
    xs = torch.rand((M, scale_k), dtype=dtypes.fp32, device="cuda")
    ws = torch.rand((scale_n, scale_k), dtype=dtypes.fp32, device="cuda")
    if mode == "stability":
        xs = xs * 1.5 + 0.125
        ws = ws * 1.5 + 0.125
    out = torch.empty((M, N), dtype=dtypes.bf16, device="cuda")
    return x, w, xs, ws, out


def run_ck(x, w, xs, ws, out):
    global _BUILD_STAMP_WRITTEN
    result = aiter.gemm_a8w8_blockscale_tune(
        x, w, xs, ws, out, EDITABLE_KERNEL_ID, SPLIT_K
    )
    if not _BUILD_STAMP_WRITTEN:
        BUILD_STAMP.parent.mkdir(parents=True, exist_ok=True)
        BUILD_STAMP.write_text(SOURCE_HASH + "\n")
        _BUILD_STAMP_WRITTEN = True
    return result


def torch_reference(x, w, xs, ws):
    M, K = x.shape
    N = w.shape[0]
    scale_k = K // BLOCK_K
    xf = (
        x.float().reshape(M, scale_k, BLOCK_K) * xs.unsqueeze(-1)
    ).reshape(M, K)
    wf_scale = (
        ws.repeat_interleave(BLOCK_N, dim=0)
        .repeat_interleave(BLOCK_K, dim=1)[:N, :K]
    )
    wf = w.float() * wf_scale
    return F.linear(xf, wf).to(dtypes.bf16)


def correctness(M: int, N: int, K: int, mode: str) -> int:
    x, w, xs, ws, out = make_inputs(M, N, K, mode)
    got = run_ck(x, w, xs, ws, out)
    torch.cuda.synchronize()
    ref = torch_reference(x, w, xs, ws)
    got_f, ref_f = got.float(), ref.float()
    noise = torch.linalg.vector_norm(got_f - ref_f).item()
    signal = torch.linalg.vector_norm(ref_f).item()
    snr = 999.0 if noise == 0 else 20.0 * math.log10(max(signal, 1e-30) / noise)
    max_diff = (got_f - ref_f).abs().max().item()
    close = torch.allclose(got_f, ref_f, rtol=2e-2, atol=2e-2)
    print(f"SNR: {snr:.4f} dB")
    print(f"allclose: {close}")
    print(f"max_diff: {max_diff:.8e}")
    return 0 if close and math.isfinite(snr) else 1


def benchmark(M: int, N: int, K: int, warmup: int, iters: int) -> int:
    x, w, xs, ws, out = make_inputs(M, N, K, "")
    for _ in range(max(1, warmup)):
        run_ck(x, w, xs, ws, out)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run_ck(x, w, xs, ws, out)
        ends[i].record()
    torch.cuda.synchronize()
    times = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    for value in times:
        print(f"wall_ms: {value:.8f}")
    median = statistics.median(times)
    print(f"median_ms: {median:.8f}")
    print(f"case_ms: primary {median:.8f}")
    return 0


def profile_run(M: int, N: int, K: int) -> int:
    x, w, xs, ws, out = make_inputs(M, N, K, "")
    for _ in range(3):
        run_ck(x, w, xs, ws, out)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_ck(x, w, xs, ws, out)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    print(f"wall_ms: {elapsed:.8f}")
    print(f"median_ms: {elapsed:.8f}")
    print(f"case_ms: primary {elapsed:.8f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="default")
    parser.add_argument("--mode", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", default="")
    args = parser.parse_args()
    M, N, K = parse_shape(args.shape)
    if args.profile_run:
        return profile_run(M, N, K)
    if args.bench_mode:
        return benchmark(M, N, K, args.warmup, args.iters)
    return correctness(M, N, K, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
