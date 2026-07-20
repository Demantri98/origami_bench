#!/usr/bin/env python3
"""Select a concrete aiter CK kernel per dataset shape via Origami.

For each (M, N, K[, q_dtype_w]) this builds an Origami ``problem_t`` and scores
candidate CK configs, returning the best. Two candidate sources:

  * ``--candidates aiter`` (default, the linchpin): one ``config_t`` per REAL
    aiter ``kernelId`` (from ``tools/ck_kernel_map.py``), scored with
    ``compute_total_latency``. The winner is therefore a concrete aiter CK
    instance -> emits ``best_kernelId`` + ``best_kernelName`` that
    ``aiter_run.py`` can execute directly as a backup heuristic.
  * ``--candidates grid``: an abstract tile grid (no aiter needed). Emits tile
    sizes only (no kernelId). Useful as a dtype/hardware sanity check.

Origami's predicted latency is in GPU cycles -- a RANKING signal, not a wall
time. It selects/ranks; it does not benchmark.

ENVIRONMENT: needs ``import origami`` (py3.12 wheel in the origami-dev
container). ``--candidates aiter`` additionally imports aiter's pure-python
kernel tables (no torch). Off a ROCm/origami box the tool writes an
``available=false`` marker row and exits 0 (safe dry-run on the host).

  docker exec origami-dev bash -lc 'cd /home/demantri/ck_origami_bench && \
    python3 tools/origami_select.py --family a8w8_blockscale --gpu-target gfx950'
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.families import FAMILIES, TARGET_GFX, Family, get_family  # noqa: E402
import ck_kernel_map  # noqa: E402

# Per-arch analytical hardware specs (N_CU, lds, rf, L2, clock_khz), matching the
# Origami test fixtures. Used only when no live device is available.
_ARCH_SPECS: dict[str, tuple[int, int, int, int, int]] = {
    "gfx90a": (110, 64 * 1024, 512 * 1024, 8 * 1024 * 1024, 1_700_000),
    "gfx942": (228, 64 * 1024, 512 * 1024, 24 * 1024 * 1024, 1_700_000),
    "gfx950": (304, 64 * 1024, 512 * 1024, 32 * 1024 * 1024, 2_100_000),
    "gfx1100": (96, 64 * 1024, 512 * 1024, 6 * 1024 * 1024, 2_500_000),
    "gfx1200": (32, 128 * 1024, 512 * 1024, 4 * 1024 * 1024, 2_700_000),
    "gfx1201": (60, 128 * 1024, 512 * 1024, 6 * 1024 * 1024, 2_500_000),
}

# Abstract tile grid (only for --candidates grid).
_GRID = [{"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "waves_per_eu": occ}
         for bm in [16, 32, 64, 128, 256]
         for bn in [16, 32, 64, 128, 256, 512]
         for bk in [64, 128, 256, 512]
         for occ in [1, 2]]

_FIELDS = ["family", "gfx", "M", "N", "K", "q_dtype_w", "available",
           "hardware_source", "cu_num", "candidates",
           "best_kernelId", "best_kernelName",
           "best_BLOCK_M", "best_BLOCK_N", "best_BLOCK_K", "best_waves_per_eu",
           "mi_m", "mi_n", "mi_k", "splitK",
           "pred_latency_cycles", "pred_tflops", "topk_json", "error"]


def _select_rows(rows, limit, sample, sample_seed):
    """Deterministic row subset. --sample draws a representative random subset
    (same seed => same rows across tools, so joins line up); --limit takes the
    first N (fast debug). --sample wins if both are set."""
    import random
    if sample and sample < len(rows):
        idx = sorted(random.Random(sample_seed).sample(range(len(rows)), sample))
        return [rows[i] for i in idx]
    if limit:
        return rows[:limit]
    return rows


def _dt(origami, tok, fb="f8"):
    for t in (tok, fb):
        try:
            return origami.string_to_datatype(t)
        except Exception:
            continue
    return origami.string_to_datatype("f16")


def _resolve_hardware(origami, gpu_target, prefer_device, device_index):
    if prefer_device:
        try:
            return origami.get_hardware_for_device(device_index), f"device:{device_index}"
        except Exception:
            pass
    arch_enum = getattr(origami.architecture_t, gpu_target, None)
    spec = _ARCH_SPECS.get(gpu_target)
    if arch_enum is None or spec is None:
        raise ValueError(f"no analytical spec for '{gpu_target}' (known: {sorted(_ARCH_SPECS)})")
    return origami.get_hardware_for_arch(arch_enum, *spec), f"arch:{gpu_target}"


def _make_problem(origami, m, n, k, od):
    p = origami.problem_t()
    p.size = origami.dim3_t(m, n, k)
    p.batch = 1
    p.a_transpose = origami.transpose_t.T
    p.b_transpose = origami.transpose_t.N
    p.a_dtype = _dt(origami, od["a"])
    p.b_dtype = _dt(origami, od["b"])
    p.d_dtype = _dt(origami, od["out"], "bf16")
    p.c_dtype = p.d_dtype
    p.mi_dtype = _dt(origami, od["mi"], od["a"])
    p.a_mx_block_size = od.get("mx", 0)
    p.b_mx_block_size = od.get("mx", 0)
    return p


def _compute_splitk(m, n, k, tm, tn, tk, cu_num):
    if min(tm, tn, tk, cu_num) <= 0:
        return 0
    tile_num = ((m + tm - 1) // tm) * ((n + tn - 1) // tn)
    if tile_num <= 0:
        return 0
    cus_per_tile = cu_num / tile_num
    sk = 0
    while cus_per_tile >= 2 ** (sk + 1) and (2 ** (sk + 1) * tk) < 2 * k:
        sk += 1
    return sk


def _grid_configs(origami, mi, grid):
    configs = []
    for c in grid:
        try:
            cfg = origami.config_t()
            cfg.mt = origami.dim3_t(int(c["BLOCK_M"]), int(c["BLOCK_N"]), int(c["BLOCK_K"]))
            cfg.mi = mi
            cfg.occupancy = int(c.get("waves_per_eu", 2))
        except Exception:
            continue
        configs.append((None, cfg))
    return configs


def process_family(fam, datasets_dir, out_dir, gpu_target, prefer_device,
                   device_index, topk, limit, candidates, occupancy,
                   sample=None, sample_seed=0):
    ds = fam.dataset_path(datasets_dir)
    if not os.path.exists(ds):
        raise FileNotFoundError(f"dataset missing: {ds} (run gen_datasets.py)")
    rows = list(csv.DictReader(open(ds, newline="")))
    rows = _select_rows(rows, limit, sample, sample_seed)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fam.key}_origami.csv")

    try:
        import origami
    except Exception as e:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_FIELDS)
            w.writeheader()
            w.writerow({"family": fam.key, "gfx": gpu_target, "available": False,
                        "candidates": candidates, "error": f"origami import failed: {e}"})
        print(f"  [{fam.key}] origami UNAVAILABLE here ({e}); wrote marker -> {out_path}")
        return out_path

    hardware, hw_source = _resolve_hardware(origami, gpu_target, prefer_device, device_index)
    cu_num = int(getattr(hardware, "N_CU", 0) or 0)
    ktable = ck_kernel_map.load_kernel_table(fam) if candidates == "aiter" else {}

    # Cache candidate config lists per dtype token (shape-independent).
    cfg_cache: dict[str, list] = {}

    def _configs_for(dt_tok, od):
        if dt_tok in cfg_cache:
            return cfg_cache[dt_tok]
        if candidates == "aiter":
            cfgs = ck_kernel_map.build_configs(origami, hardware, fam, dt_tok, occupancy)
        else:
            mi = hardware.get_recommended_matrix_instruction(_dt(origami, od["mi"], od["a"]))
            cfgs = _grid_configs(origami, mi, _GRID)
        cfg_cache[dt_tok] = cfgs
        return cfgs

    n_ok = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
            dt_tok = r.get("q_dtype_w", "") if fam.has_dtype else ""
            od = fam.origami_dtypes.get(dt_tok) or next(iter(fam.origami_dtypes.values()))
            rec = {"family": fam.key, "gfx": gpu_target, "M": m, "N": n, "K": k,
                   "q_dtype_w": dt_tok, "available": True, "hardware_source": hw_source,
                   "cu_num": cu_num, "candidates": candidates}
            try:
                configs = _configs_for(dt_tok, od)
                problem = _make_problem(origami, m, n, k, od)
                scored = ck_kernel_map.rank_kernelids(origami, hardware, problem, configs)
                if not scored:
                    rec["error"] = "no config scored"
                    w.writerow(rec)
                    continue
                best_kid, best_lat = scored[0]
                # Resolve tile / mi / name from the winning config.
                cfg_by_kid = {kid: cfg for kid, cfg in configs}
                bcfg = cfg_by_kid.get(best_kid)
                bm, bn, bk = int(bcfg.mt.m), int(bcfg.mt.n), int(bcfg.mt.k)
                mi_m, mi_n, mi_k = int(bcfg.mi.m), int(bcfg.mi.n), int(bcfg.mi.k)
                name = ktable[best_kid].name if (candidates == "aiter" and best_kid in ktable) else ""
                try:
                    tflops = float(origami.compute_perf_gflops(hardware, problem, best_lat)) / 1000.0
                except Exception:
                    tflops = ""
                topk_rows = []
                for kid, lat in scored[: max(1, topk)]:
                    entry = {"kernelId": kid, "latency_cycles": round(lat, 2)}
                    if candidates == "aiter" and kid in ktable:
                        entry["kernelName"] = ktable[kid].name
                    topk_rows.append(entry)
                rec.update({
                    "best_kernelId": best_kid if best_kid is not None else "",
                    "best_kernelName": name,
                    "best_BLOCK_M": bm, "best_BLOCK_N": bn, "best_BLOCK_K": bk,
                    "best_waves_per_eu": int(getattr(bcfg, "occupancy", 0)),
                    "mi_m": mi_m, "mi_n": mi_n, "mi_k": mi_k,
                    # Origami selects the kernel/tile only; it does not model
                    # splitK. Emit 0 (same as aiter's default fallback on a miss)
                    # so the comparison isolates kernel choice. A separate
                    # (validated) splitK search is future work -- an ad-hoc splitK
                    # heuristic degrades correctness for a8w8/a4w4 kernels.
                    "splitK": 0,
                    "pred_latency_cycles": round(best_lat, 2),
                    "pred_tflops": tflops,
                    "topk_json": json.dumps(topk_rows),
                })
                n_ok += 1
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            w.writerow(rec)
    print(f"  [{fam.key}] selected {n_ok}/{len(rows)} shapes "
          f"(candidates={candidates}, {hw_source}) -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*", default=list(FAMILIES))
    ap.add_argument("--datasets-dir", default=os.path.join(_ROOT, "datasets"))
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "results"))
    ap.add_argument("--gpu-target", default=TARGET_GFX)
    ap.add_argument("--candidates", choices=["aiter", "grid"], default="aiter",
                    help="'aiter': rank real kernelIds (linchpin); 'grid': abstract tiles")
    ap.add_argument("--occupancy", type=int, default=2,
                    help="constant occupancy for CK instances (not encoded in the table)")
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="first N rows (debug)")
    ap.add_argument("--sample", type=int, default=None,
                    help="random representative subset of N rows (use same value in aiter_run)")
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Origami CK kernel selection (target={args.gpu_target}, candidates={args.candidates})\n")
    for key in args.family:
        fam = get_family(key)
        try:
            process_family(fam, args.datasets_dir, args.out_dir, args.gpu_target,
                            not args.no_device, args.device_index, args.topk, args.limit,
                            args.candidates, args.occupancy, args.sample, args.sample_seed)
        except Exception as e:
            print(f"  [{key}] FAILED: {e}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
