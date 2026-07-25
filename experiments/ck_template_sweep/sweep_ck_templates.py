#!/usr/bin/env python3
"""
Sweep a chosen (M, N, K) GEMM across ALL CK a8w8-blockscale candidate templates
via the aiter tune op, and report per-template latency / TFLOPS / correctness.

This gives a direct CK baseline (the full template pool aiter's tuner picks from)
to compare an Origami-selected kernel against.

Run inside the ROCm container, e.g.:
  docker exec -e PYTHONPATH=/home/demantri/aiter origami-dev \
    python3 /home/demantri/origami_bench/experiments/ck_template_sweep/sweep_ck_templates.py --shape 2048,2048,3072
"""
import argparse
import csv
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
from aiter.ops.gemm_op_common import get_padded_m
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx

sys.path.insert(0, "/home/demantri/aiter/csrc/ck_gemm_a8w8_blockscale")
from gemm_a8w8_blockscale_instance import (  # noqa: E402
    candidate_kernels_dict,
    candidate_kernels_by_name,
)

BLOCK_SHAPE = (128, 128)
INVALID = float("inf")


def gen_data(M, N, K, seed=0, device="cuda"):
    torch.manual_seed(seed)
    bn, bk = BLOCK_SHAPE
    scale_n = (N + bn - 1) // bn
    scale_k = (K + bk - 1) // bk
    x = (torch.rand((M, K), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    w = (torch.rand((N, K), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    x_scale = torch.rand([M, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    out = torch.empty(M, N, dtype=dtypes.bf16, device=device)
    return x, w, x_scale, w_scale, out


def ref_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    bn, bk = BLOCK_SHAPE
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + bn - 1) // bn
    scale_k = (k + bk - 1) // bk
    xf = x.to(x_scale.dtype).view(m, k // bk, bk) * x_scale.unsqueeze(-1)
    xf = xf.view(m, k)
    ws = rearrange(
        w_scale.view(-1, 1).repeat(1, bn * bk).view(scale_n, scale_k, bn, bk),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )[:n, :k]
    wf = weight.to(ws.dtype) * ws
    return F.linear(xf.to(dtypes.fp32), wf.to(dtypes.fp32)).to(dtype)


def tile_str(k):
    return f"{k.MPerBLOCK}x{k.NPerBLOCK}x{k.KPerBLOCK}"


def prod_choice(M, N, K):
    """What aiter's production dispatch would pick for this exact (M,N,K)."""
    cfg = get_CKGEMM_config(M, N, K, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE)
    if cfg is None or cfg.get("libtype") != "ck":
        return None, cfg
    name = cfg.get("kernelName", "")
    k = candidate_kernels_by_name.get(name)
    # map name -> candidate id
    kid = next((i for i, kk in candidate_kernels_dict.items() if kk.name == name), None)
    return kid, cfg


def bench_shape(M, N, K, args):
    x, w, xs, ws, out = gen_data(M, N, K, args.seed)
    ref = ref_torch(x, w, xs, ws) if not args.no_check else None
    ref_abs = ref.float().abs().mean().item() + 1e-9 if ref is not None else None

    prod_kid, prod_cfg = prod_choice(M, N, K)
    default_name = (
        "a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_"
        "16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1"
    )
    default_kid = next(
        (i for i, kk in candidate_kernels_dict.items() if kk.name == default_name), None
    )

    rows = []
    for kid in sorted(candidate_kernels_dict):
        k = candidate_kernels_dict[kid]
        if args.splitk:
            max_sk = aiter.compute_gemm_SplitK(
                M, N, K, k.MPerBLOCK, k.NPerBLOCK, k.KPerBLOCK
            )
            sk_list = list(range(min(max_sk, args.max_splitk) + 1))
        else:
            sk_list = [0]

        best_us, best_sk, best_err, status = INVALID, None, None, "unsupported"
        for sk in sk_list:
            try:
                o, us = run_perftest(
                    aiter.gemm_a8w8_blockscale_tune,
                    x, w, xs, ws, out, kid, sk,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    num_rotate_args=1,
                    use_cuda_event=True,
                )
                torch.cuda.synchronize()
                if ref is not None:
                    err = (o.float() - ref.float()).abs().mean().item() / ref_abs
                else:
                    err = float("nan")
                if us < best_us:
                    best_us, best_sk, best_err, status = us, sk, err, "ok"
            except Exception:
                continue

        tflops = (2 * M * N * K / (best_us * 1e-6) / 1e12) if status == "ok" else 0.0
        rows.append(
            {
                "kernelId": kid,
                "tile_MxNxK": tile_str(k),
                "xdl": f"{k.MPerXDL}x{k.NPerXDL}",
                "wave": f"{k.WAVE_MAP_M}x{k.WAVE_MAP_N}",
                "pipe": f"{k.PIPELINE_Sched.lower()}_v{k.PIPELINE_VERSION}",
                "splitK": best_sk if best_sk is not None else "-",
                "us": round(best_us, 3) if status == "ok" else "",
                "tflops": round(tflops, 1) if status == "ok" else "",
                "errRatio": round(best_err, 5) if status == "ok" else "",
                "status": status,
                "name": k.name,
            }
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    ok_rows.sort(key=lambda r: r["us"])
    return rows, ok_rows, prod_kid, prod_cfg, default_kid


def print_shape(M, N, K, rows, ok_rows, prod_kid, prod_cfg, default_kid, args):
    pm0 = get_padded_m(M, N, K, 0)
    pm1 = get_padded_m(M, N, K, 1)
    print("\n" + "=" * 108)
    print(
        f"SHAPE (M,N,K)=({M},{N},{K})   padded_M[gl0]={pm0} padded_M[gl1]={pm1}   "
        f"gfx={get_gfx()} cu={get_cu_num()}"
    )
    if prod_cfg is None:
        print(f"  production dispatch: FALLBACK -> default/backup CK kernel (kernelId {default_kid})")
    else:
        print(
            f"  production dispatch: kernelId {prod_kid} "
            f"[{candidate_kernels_dict[prod_kid].MPerBLOCK}x{candidate_kernels_dict[prod_kid].NPerBLOCK}x{candidate_kernels_dict[prod_kid].KPerBLOCK}]"
            f" libtype={prod_cfg.get('libtype')} splitK={prod_cfg.get('splitK')}"
        )
    print("-" * 108)
    hdr = f"{'kid':>3} {'tile MxNxK':>13} {'xdl':>6} {'wave':>5} {'pipe':>16} {'splitK':>6} {'us':>9} {'TFLOPS':>7} {'errRatio':>9} {'status':>11}"
    print(hdr)
    print("-" * 108)
    ranked = ok_rows + [r for r in rows if r["status"] != "ok"]
    for rank, r in enumerate(ranked):
        marks = ""
        if r["kernelId"] == prod_kid:
            marks += " <prod"
        if r["kernelId"] == default_kid:
            marks += " <default"
        best = " *BEST" if (r["status"] == "ok" and rank == 0) else ""
        print(
            f"{r['kernelId']:>3} {r['tile_MxNxK']:>13} {r['xdl']:>6} {r['wave']:>5} "
            f"{r['pipe']:>16} {str(r['splitK']):>6} {str(r['us']):>9} {str(r['tflops']):>7} "
            f"{str(r['errRatio']):>9} {r['status']:>11}{best}{marks}"
        )
    if ok_rows:
        b = ok_rows[0]
        print("-" * 108)
        print(
            f"  BEST template: kernelId {b['kernelId']} tile {b['tile_MxNxK']} "
            f"splitK={b['splitK']} -> {b['us']} us, {b['tflops']} TFLOPS"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shape", action="append", default=None,
        help="M,N,K (repeatable). Default: a small/medium/large set at N,K=2048,3072",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splitk", action="store_true", help="sweep splitK per template and keep best")
    ap.add_argument("--max-splitk", type=int, default=4, dest="max_splitk")
    ap.add_argument("--no-check", action="store_true", help="skip torch correctness check (faster)")
    ap.add_argument("--out-csv", default="/home/demantri/origami_bench/experiments/ck_template_sweep/ck_template_sweep.csv")
    args = ap.parse_args()

    if args.shape:
        shapes = [tuple(int(v) for v in s.split(",")) for s in args.shape]
    else:
        shapes = [(256, 2048, 3072), (2048, 2048, 3072), (8192, 2048, 3072)]

    print(f"Sweeping {len(candidate_kernels_dict)} CK candidate templates over {len(shapes)} shape(s)")
    print(f"timing: warmup={args.warmup} iters={args.iters} splitK_sweep={args.splitk} check={not args.no_check}")

    all_rows = []
    for (M, N, K) in shapes:
        rows, ok_rows, prod_kid, prod_cfg, default_kid = bench_shape(M, N, K, args)
        print_shape(M, N, K, rows, ok_rows, prod_kid, prod_cfg, default_kid, args)
        for r in rows:
            all_rows.append({"M": M, "N": N, "K": K, **r})

    with open(args.out_csv, "w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=["M", "N", "K", "kernelId", "tile_MxNxK", "xdl", "wave",
                        "pipe", "splitK", "us", "tflops", "errRatio", "status", "name"],
        )
        wcsv.writeheader()
        wcsv.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
