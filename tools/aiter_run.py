#!/usr/bin/env python3
"""Run + benchmark aiter's CK dense GEMM kernels over the novel-shape datasets.

For each (M, N, K[, q_dtype_w]) in a family's dataset this tool:

  * builds valid input tensors exactly as aiter's own tune scripts do
    (correct dtype / scale layout / weight preshuffle per family),
  * MODE "prod" (default): calls the high-level aiter op -- i.e. the real
    production dispatch, which for these untuned shapes falls back to the C++
    heuristic / default CK instance -- and times it with aiter's own
    ``run_perftest``; records us, TFLOPS, bandwidth, and whether a tuned CSV row
    existed (it should not, by construction),
  * MODE "sweep" (optional): iterates every CK kernelId (and splitK) via the
    family's ``*_tune`` entrypoint, benchmarks each, and records aiter's
    EMPIRICAL best kernelId / kernelName / us. This is the "oracle" aiter can
    reach, to compare against both the prod fallback and Origami's prediction.

ENVIRONMENT: host Python 3.9 with aiter importable and a gfx950 GPU visible.

  python3 tools/aiter_run.py --family a8w8_blockscale --mode prod --limit 50
  python3 tools/aiter_run.py --family a4w4_blockscale --mode both --limit 20

Outputs results/<family>_aiter_<mode>.csv. Every shape is wrapped in try/except
so one bad shape never aborts the run.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.families import (  # noqa: E402
    DT_INT8, FAMILIES, TARGET_GFX, Family, get_family,
)
import ck_kernel_map  # noqa: E402

import torch  # noqa: E402
import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402


def _time_us(fn, warmup, iters):
    """Device time per call in microseconds via hipEvents.

    Direct event timing avoids the torch-profiler overhead that dominates
    aiter's ``run_perftest`` when sweeping many shapes, while measuring the same
    thing (GPU kernel time). ``fn`` is a zero-arg callable running one GEMM.
    """
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # ms/iter -> us/iter


def _err_ratio(a, b):
    """Fraction of elements where a and b are NOT close (rtol/atol from caller).

    NaN/Inf count as mismatches (torch.isclose treats them as not-close), so a
    garbage kernel yields err_ratio≈1.
    """
    a = a.detach().float()
    b = b.detach().float()
    if a.shape != b.shape:
        return 1.0
    close = torch.isclose(a, b, rtol=_CHK_RTOL, atol=_CHK_ATOL)
    return float(1.0 - close.float().mean().item())


def _run_once_output(sweep, kernel_id, split_k, m, n):
    """Run one Origami-picked kernel and return its output sliced to [:m, :n].

    The output is zeroed first: with splitK/KBatch>1 the CK kernel accumulates
    into the output (atomic add across K-splits), so a dirty buffer yields wrong
    results. aiter's high-level op zeroes internally; the low-level tune
    entrypoint expects the caller to. (Zeroing only matters for correctness, not
    timing, so the timing path leaves it alone.)
    """
    tune_fn = sweep["tune_fn"]
    xq, wq, xs, ws, out = sweep["args"]
    out.zero_()
    tune_fn(xq, wq, xs, ws, out, int(kernel_id), int(split_k))
    torch.cuda.synchronize()
    return out[:m, :n]


# Correctness tolerances (set from CLI). The gate compares the Origami-picked
# kernel's output against aiter's trusted production output on identical inputs.
_CHK_RTOL = 1e-2
_CHK_ATOL = 1e-2


def _tflops_bw(m, n, k, us, bpes):
    if not us or us <= 0:
        return "", ""
    flop = m * n * k * 2
    tflops = round(flop / (us * 1e6), 2)
    inb, wb, outb = bpes
    bw = round((m * k * inb + n * k * wb + m * n * outb) / (us * 1e-6) / 1e9, 2)
    return tflops, bw


# --------------------------------------------------------------------------- #
# Per-family tensor builders. Each returns a dict:
#   prod:  zero-arg callable running the high-level aiter op (production path)
#   sweep: optional dict {tune_fn, kernels_list, args:(XQ,WQ,xs,ws,Out)}
# --------------------------------------------------------------------------- #
def _fp8():
    return dtypes.fp8


def _iter_kernel_ids(kernels_list):
    """kernels_list is either a list or an index-keyed dict."""
    if isinstance(kernels_list, dict):
        return list(kernels_list.keys())
    return list(range(len(kernels_list)))


def _kernel_attrs(kernels_list, kid):
    k = kernels_list[kid]
    return (getattr(k, "name", str(kid)),
            getattr(k, "MPerBLOCK", 0),
            getattr(k, "NPerBLOCK", 0),
            getattr(k, "KPerBLOCK", 0))


def build_a8w8(m, n, k, dt, device="cuda"):
    if dt == DT_INT8:
        x = torch.randint(-20, 20, (m, k), dtype=dtypes.i8, device=device)
        w = torch.randint(-20, 20, (n, k), dtype=dtypes.i8, device=device)
        x_scale = torch.rand([m, 1], dtype=dtypes.bf16, device=device)
        w_scale = torch.rand([1, n], dtype=dtypes.bf16, device=device)
    else:
        xf = torch.randn((m, k), dtype=dtypes.bf16, device=device)
        wf = torch.randn((n, k), dtype=dtypes.bf16, device=device)
        x, x_scale = aiter.pertoken_quant(xf, quant_dtype=_fp8())
        w, w_scale = aiter.pertoken_quant(wf, quant_dtype=_fp8())
    Out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    return {
        "prod": lambda: aiter.gemm_a8w8(x, w, x_scale, w_scale, dtype=dtypes.bf16),
        "sweep": {"tune_fn": aiter.gemm_a8w8_tune, "args": (x, w, x_scale, w_scale, Out)},
    }


def build_a8w8_bpreshuffle(m, n, k, dt, device="cuda"):
    xf = torch.randn((m, k), dtype=dtypes.bf16, device=device)
    wf = torch.randn((n, k), dtype=dtypes.bf16, device=device)
    x, x_scale = aiter.pertoken_quant(xf, quant_dtype=_fp8())
    w_raw, w_scale = aiter.pertoken_quant(wf, quant_dtype=_fp8())
    w = shuffle_weight(w_raw, layout=(16, 16))
    Out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    return {
        "prod": lambda: aiter.gemm_a8w8_bpreshuffle(x, w, x_scale, w_scale, dtype=dtypes.bf16),
        "sweep": {"tune_fn": aiter.gemm_a8w8_bpreshuffle_tune,
                  "args": (x, w, x_scale, w_scale, Out)},
    }


def _blockscale_tensors(m, n, k, device="cuda"):
    scale_n = (n + 127) // 128
    scale_k = (k + 127) // 128
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(_fp8())
    w = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(_fp8())
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    return x, w, x_scale, w_scale


def build_a8w8_blockscale(m, n, k, dt, device="cuda"):
    x, w, x_scale, w_scale = _blockscale_tensors(m, n, k, device)
    Out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    return {
        "prod": lambda: aiter.gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=dtypes.bf16),
        "sweep": {"tune_fn": aiter.gemm_a8w8_blockscale_tune,
                  "args": (x, w, x_scale, w_scale, Out)},
    }


def build_a8w8_blockscale_bpreshuffle(m, n, k, dt, device="cuda"):
    x, w_raw, x_scale, w_scale = _blockscale_tensors(m, n, k, device)
    w = shuffle_weight(w_raw, layout=(16, 16))
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    Out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    return {
        "prod": lambda: aiter.gemm_a8w8_blockscale_bpreshuffle(
            x, w, x_scale_t, w_scale, dtype=dtypes.bf16),
        "sweep": {"tune_fn": aiter.gemm_a8w8_blockscale_bpreshuffle_tune,
                  "args": (x, w, x_scale_t, w_scale, Out)},
    }


def build_a4w4_blockscale(m, n, k, dt, device="cuda"):
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    xf = torch.randn((m, k), dtype=dtypes.bf16, device=device)
    wf = torch.randn((n, k), dtype=dtypes.bf16, device=device)
    x, x_ss = quant_func(xf, shuffle=True)
    w, w_ss = quant_func(wf, shuffle=True)
    w_shuffle = shuffle_weight(w)
    Out = torch.empty((m + 255) // 256 * 256, n, dtype=dtypes.bf16, device=device)
    return {
        "prod": lambda: aiter.gemm_a4w4(x, w_shuffle, x_ss, w_ss),
        "sweep": {"tune_fn": aiter.gemm_a4w4_blockscale_tune,
                  "args": (x, w_shuffle, x_ss, w_ss, Out)},
    }


_BUILDERS = {
    "a8w8": build_a8w8,
    "a8w8_bpreshuffle": build_a8w8_bpreshuffle,
    "a8w8_blockscale": build_a8w8_blockscale,
    "a8w8_blockscale_bpreshuffle": build_a8w8_blockscale_bpreshuffle,
    "a4w4_blockscale": build_a4w4_blockscale,
}


def run_kernel_id(sweep: dict, kernel_id: int, split_k: int, warmup, iters):
    """Benchmark ONE explicit kernelId+splitK (single run, not a search)."""
    tune_fn = sweep["tune_fn"]
    xq, wq, xs, ws, out = sweep["args"]
    return _time_us(lambda: tune_fn(xq, wq, xs, ws, out, int(kernel_id), int(split_k)),
                    warmup, iters)


def sweep_best(sweep: dict, table, m, n, k, warmup, iters, max_kernels=None):
    """(Optional oracle) benchmark every kernelId x splitK; return best.

    ``table`` is {kernelId: KernelInfo} from ck_kernel_map.load_kernel_table.
    """
    tune_fn = sweep["tune_fn"]
    xq, wq, xs, ws, out = sweep["args"]
    best = (None, None, math.inf)
    tried = 0
    for kid, ki in table.items():
        name, (mb, nb, kb) = ki.name, ki.mt
        try:
            sk_max = aiter.compute_gemm_SplitK(m, n, k, mb, nb, kb) if mb else 0
        except Exception:
            sk_max = 0
        for sk in range(sk_max + 1):
            try:
                us = _time_us(lambda: tune_fn(xq, wq, xs, ws, out, kid, sk), warmup, iters)
            except Exception:
                continue
            if us and 0 < us < best[2]:
                best = (kid, name, us)
        tried += 1
        if max_kernels and tried >= max_kernels:
            break
    return best


def _load_origami_map(fam: Family, out_dir: str) -> dict[tuple, dict]:
    """Read results/<fam>_origami.csv -> {(M,N,K,q_dtype_w): {kernelId,splitK,name}}.

    Produced by origami_select.py --candidates aiter. Empty if not present.
    """
    path = os.path.join(out_dir, f"{fam.key}_origami.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        kid = r.get("best_kernelId", "")
        if kid in ("", None):
            continue
        key = (r.get("M"), r.get("N"), r.get("K"), r.get("q_dtype_w", ""))
        out[key] = {"kernelId": int(kid), "splitK": int(r.get("splitK") or 0),
                    "name": r.get("best_kernelName", "")}
    return out


def _select_rows(rows, limit, sample, sample_seed):
    """Match origami_select._select_rows so both tools pick identical rows."""
    import random
    if sample and sample < len(rows):
        idx = sorted(random.Random(sample_seed).sample(range(len(rows)), sample))
        return [rows[i] for i in idx]
    if limit:
        return rows[:limit]
    return rows


def process_family(fam: Family, datasets_dir, out_dir, mode, limit, warmup, iters,
                   sweep_iters, sweep_max_kernels, use_origami, sample=None, sample_seed=0,
                   check=True, check_max_err=0.02):
    ds = fam.dataset_path(datasets_dir)
    if not os.path.exists(ds):
        raise FileNotFoundError(f"dataset missing: {ds} (run gen_datasets.py)")
    rows = list(csv.DictReader(open(ds, newline="")))
    rows = _select_rows(rows, limit, sample, sample_seed)
    builder = _BUILDERS[fam.key]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fam.key}_aiter_{mode}.csv")

    # Kernel table (for the sweep oracle only); tolerate load failure.
    ktable = {}
    if mode in ("sweep", "both"):
        try:
            ktable = ck_kernel_map.load_kernel_table(fam)
        except Exception as e:
            print(f"    [{fam.key}] kernel table load failed ({e}); sweep unavailable")

    # Origami backup-heuristic picks (only meaningful in prod mode).
    origami_map = _load_origami_map(fam, out_dir) if (use_origami and mode == "prod") else {}
    if use_origami and mode == "prod" and not origami_map:
        print(f"    [{fam.key}] note: no Origami picks found "
              f"(run origami_select.py --candidates aiter first); prod-native only")

    fieldnames = ["family", "gfx", "M", "N", "K", "q_dtype_w",
                  "prod_us", "prod_tflops", "prod_bw", "prod_status",
                  "origami_kernelId", "origami_kernelName", "origami_splitK",
                  "origami_us", "origami_tflops", "origami_status",
                  "origami_err_ratio", "origami_correct",
                  "winner", "origami_speedup_vs_prod",
                  "sweep_best_kernelId", "sweep_best_kernelName", "sweep_best_us",
                  "sweep_best_tflops", "sweep_status"]

    n_ok = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
            dt = r.get("q_dtype_w", "") if fam.has_dtype else ""
            rec = {"family": fam.key, "gfx": TARGET_GFX, "M": m, "N": n, "K": k,
                   "q_dtype_w": dt}
            try:
                built = builder(m, n, k, dt)
            except Exception as e:
                rec["prod_status"] = f"build_error:{type(e).__name__}:{e}"
                w.writerow(rec)
                continue

            prod_us = None
            if mode in ("prod", "both"):
                try:
                    prod_us = _time_us(built["prod"], warmup, iters)
                    tf, bw = _tflops_bw(m, n, k, prod_us, fam.bpes)
                    rec.update({"prod_us": round(prod_us, 4), "prod_tflops": tf,
                                "prod_bw": bw, "prod_status": "ok"})
                    n_ok += 1
                except Exception as e:
                    rec["prod_status"] = f"error:{type(e).__name__}:{e}"

            # Origami backup heuristic: run the Origami-picked kernelId (single run).
            if mode == "prod" and origami_map:
                pick = origami_map.get((str(m), str(n), str(k), dt))
                sweep = built.get("sweep")
                if not pick:
                    rec["origami_status"] = "no_pick"
                elif not sweep:
                    rec["origami_status"] = "no_explicit_kernel_path"
                else:
                    try:
                        ous = run_kernel_id(sweep, pick["kernelId"], pick["splitK"],
                                            warmup, iters)
                        otf, _ = _tflops_bw(m, n, k, ous, fam.bpes)
                        rec.update({"origami_kernelId": pick["kernelId"],
                                    "origami_kernelName": pick["name"],
                                    "origami_splitK": pick["splitK"],
                                    "origami_us": round(ous, 4), "origami_tflops": otf,
                                    "origami_status": "ok"})
                        # Correctness gate: the Origami-picked kernel must match
                        # aiter's trusted production output on identical inputs.
                        correct = True
                        if check:
                            try:
                                y_prod = built["prod"]()
                                torch.cuda.synchronize()
                                y_prod = y_prod[:m, :n]
                                y_orig = _run_once_output(sweep, pick["kernelId"],
                                                          pick["splitK"], m, n)
                                er = _err_ratio(y_orig, y_prod)
                                correct = er <= check_max_err
                                rec["origami_err_ratio"] = round(er, 5)
                                rec["origami_correct"] = correct
                                if not correct:
                                    rec["origami_status"] = f"mismatch(err={er:.4f})"
                            except Exception as e:
                                correct = False
                                rec["origami_correct"] = False
                                rec["origami_status"] = f"check_error:{type(e).__name__}:{e}"
                        # Only credit a win/speedup when the pick is correct.
                        if correct and prod_us and ous and ous > 0:
                            rec["winner"] = "origami" if ous < prod_us else "prod_default"
                            rec["origami_speedup_vs_prod"] = round(prod_us / ous, 3)
                    except Exception as e:
                        rec["origami_status"] = f"error:{type(e).__name__}:{e}"

            if mode in ("sweep", "both"):
                sweep = built.get("sweep")
                if not sweep or not ktable:
                    rec["sweep_status"] = "unavailable"
                else:
                    try:
                        kid, name, us = sweep_best(sweep, ktable, m, n, k, warmup,
                                                   sweep_iters, sweep_max_kernels)
                        if kid is None:
                            rec["sweep_status"] = "no_valid_kernel"
                        else:
                            tf, _ = _tflops_bw(m, n, k, us, fam.bpes)
                            rec.update({"sweep_best_kernelId": kid,
                                        "sweep_best_kernelName": name,
                                        "sweep_best_us": round(us, 4),
                                        "sweep_best_tflops": tf, "sweep_status": "ok"})
                    except Exception as e:
                        rec["sweep_status"] = f"error:{type(e).__name__}:{e}"
            w.writerow(rec)
            fh.flush()
            if (i + 1) % 25 == 0:
                torch.cuda.empty_cache()
                print(f"    [{fam.key}] {i + 1}/{len(rows)} done")
    print(f"  [{fam.key}] mode={mode}: {n_ok}/{len(rows)} prod-ok -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*", default=list(FAMILIES))
    ap.add_argument("--datasets-dir", default=os.path.join(_ROOT, "datasets"))
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "results"))
    ap.add_argument("--mode", choices=["prod", "sweep", "both"], default="prod",
                    help="prod: native fallback (+ Origami backup pick if available). "
                         "sweep: optional exhaustive oracle.")
    ap.add_argument("--no-origami", action="store_true",
                    help="in prod mode, do NOT also run the Origami-picked kernel")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (debug)")
    ap.add_argument("--sample", type=int, default=None,
                    help="random representative subset (use same value+seed as origami_select)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100, help="timing iters/shape")
    ap.add_argument("--sweep-iters", type=int, default=20, help="timing iters/kernel in sweep")
    ap.add_argument("--sweep-max-kernels", type=int, default=None,
                    help="cap kernelIds tried per shape in sweep (debug/speed)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the correctness gate (compare Origami output vs prod)")
    ap.add_argument("--check-rtol", type=float, default=1e-2)
    ap.add_argument("--check-atol", type=float, default=1e-2)
    ap.add_argument("--check-max-err", type=float, default=0.02,
                    help="max fraction of mismatched elements to still count as correct")
    args = ap.parse_args()

    global _CHK_RTOL, _CHK_ATOL
    _CHK_RTOL, _CHK_ATOL = args.check_rtol, args.check_atol

    gfx = None
    for _getter in ("get_gfx_runtime",):
        try:
            from aiter.jit.utils.chip_info import get_gfx_runtime
            gfx = get_gfx_runtime()
        except Exception:
            pass
    print(f"aiter GEMM run (device gfx={gfx}, mode={args.mode}, "
          f"origami_backup={not args.no_origami and args.mode == 'prod'})\n")
    for key in args.family:
        fam = get_family(key)
        try:
            process_family(fam, args.datasets_dir, args.out_dir, args.mode, args.limit,
                            args.warmup, args.iters, args.sweep_iters, args.sweep_max_kernels,
                            use_origami=not args.no_origami,
                            sample=args.sample, sample_seed=args.sample_seed,
                            check=not args.no_check, check_max_err=args.check_max_err)
        except Exception as e:
            print(f"  [{key}] FAILED: {e}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
