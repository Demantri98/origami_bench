#!/usr/bin/env python3
"""The linchpin: map aiter CK kernel instances <-> Origami config_t.

Each aiter CK dense-GEMM family ships a pure-python table of kernel instances
(``kernels_list`` / ``candidate_kernels_dict`` in ``csrc/<fam>/*_common*.py``),
keyed by ``kernelId``. Every instance encodes a macrotile
(``MPerBLOCK,NPerBLOCK,KPerBLOCK``), an MFMA size (``WAVE_TILE_{M,N}`` for a8w8
rowwise, else ``MPerXDL/NPerXDL``) and a wave mapping (``WAVE_MAP_{M,N}``).

Origami's ``config_t`` is exactly (macrotile ``mt``, matrix-instruction ``mi``,
``occupancy``). So we can build ONE ``config_t`` per real ``kernelId`` and let
Origami score them with ``compute_total_latency`` -- the argmin is Origami's
recommended kernel AND a concrete aiter CK instance, with no fuzzy tile matching.
This closes the CK ``kernelId`` <-> ``config_t`` loop.

The kernel tables are torch-free, so ``load_kernel_table`` and the CLI table dump
run anywhere. ``rank_kernelids`` additionally needs ``import origami`` (available
in the origami-dev container).

CLI (dump the kernel table for inspection / offline joins):
  python3 tools/ck_kernel_map.py --family a8w8_blockscale
  # -> results/a8w8_blockscale_ck_kernels.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.families import AITER_ROOT, FAMILIES, Family, get_family  # noqa: E402


@dataclass
class KernelInfo:
    kernel_id: int
    name: str
    mt: tuple[int, int, int]      # (MPerBLOCK, NPerBLOCK, KPerBLOCK) macrotile
    mi_mn: tuple[int, int]        # MFMA (m, n)
    waves: tuple[int, int]        # (WAVE_MAP_M, WAVE_MAP_N)
    block_size: int
    pipeline_version: int


def _first_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def load_kernel_table(fam: Family) -> dict[int, KernelInfo]:
    """Import the family's csrc common module and extract its kernel table.

    Pure-python: no torch / aiter / GPU required.
    """
    subdir = os.path.join(AITER_ROOT, "csrc", fam.csrc_subdir)
    if subdir not in sys.path:
        sys.path.insert(0, subdir)
    mod = __import__(fam.kernel_module)
    raw = getattr(mod, fam.kernel_dict_attr)
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    table: dict[int, KernelInfo] = {}
    for kid, inst in items:
        mi_m = _first_attr(inst, ("WAVE_TILE_M", "MPerXDL"), 0)
        mi_n = _first_attr(inst, ("WAVE_TILE_N", "NPerXDL"), 0)
        table[int(kid)] = KernelInfo(
            kernel_id=int(kid),
            name=getattr(inst, "name", str(kid)),
            mt=(int(inst.MPerBLOCK), int(inst.NPerBLOCK), int(inst.KPerBLOCK)),
            mi_mn=(int(mi_m), int(mi_n)),
            waves=(int(_first_attr(inst, ("WAVE_MAP_M",), 0)),
                   int(_first_attr(inst, ("WAVE_MAP_N",), 0))),
            block_size=int(_first_attr(inst, ("BLOCK_SIZE",), 0)),
            pipeline_version=int(_first_attr(inst, ("PIPELINE_VERSION",), 0)),
        )
    return table


def _mi_k_for(origami, hardware, dtype_tok: str, mi_m: int, mi_n: int) -> int:
    """K of the MFMA for this dtype whose (m, n) matches the kernel's MFMA.

    Falls back to the hardware-recommended MI's K, then a sane constant.
    """
    def _dt(tok, fb="f8"):
        for t in (tok, fb):
            try:
                return origami.string_to_datatype(t)
            except Exception:
                continue
        return origami.string_to_datatype("f16")

    dt = _dt(dtype_tok)
    try:
        for mi in hardware.get_valid_matrix_instructions(dt):
            if int(mi.m) == mi_m and int(mi.n) == mi_n:
                return int(mi.k)
    except Exception:
        pass
    try:
        return int(hardware.get_recommended_matrix_instruction(dt).k)
    except Exception:
        return 32


def build_configs(origami, hardware, fam: Family, dtype_tok: str,
                  occupancy: int = 2) -> list[tuple[int, object]]:
    """One Origami ``config_t`` per real aiter kernelId.

    Returns ``[(kernelId, config_t)]``. ``occupancy`` is not encoded in the CK
    instance table (it is a compiler/launch property), so it is held constant so
    it does not bias the ranking across instances.
    """
    od = fam.origami_dtypes.get(dtype_tok) or next(iter(fam.origami_dtypes.values()))
    table = load_kernel_table(fam)
    out = []
    for kid, ki in table.items():
        try:
            cfg = origami.config_t()
            cfg.mt = origami.dim3_t(*ki.mt)
            mi_k = _mi_k_for(origami, hardware, od["mi"], ki.mi_mn[0], ki.mi_mn[1])
            cfg.mi = origami.dim3_t(ki.mi_mn[0], ki.mi_mn[1], mi_k)
            cfg.occupancy = occupancy
        except Exception:
            continue
        out.append((kid, cfg))
    return out


def rank_kernelids(origami, hardware, problem, configs) -> list[tuple[int, float]]:
    """Score each (kernelId, config) with compute_total_latency; sort ascending.

    Using ``compute_total_latency`` per config (rather than ``rank_configs``)
    keeps an exact kernelId<->latency pairing, since multiple kernelIds can map
    to the same Origami-visible config.
    """
    import math
    n_cu = getattr(hardware, "N_CU", 0)
    scored = []
    for kid, cfg in configs:
        try:
            lat = float(origami.compute_total_latency(problem, hardware, cfg, n_cu))
        except Exception:
            continue
        # Skip infeasible configs: Origami returns a huge sentinel (~DBL_MAX)
        # when a tile does not fit / is invalid for the problem+hardware.
        if math.isfinite(lat) and 0 < lat < 1e300:
            scored.append((kid, lat))
    scored.sort(key=lambda kl: kl[1])
    return scored


def dump_table(fam: Family, out_dir: str) -> str:
    table = load_kernel_table(fam)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fam.key}_ck_kernels.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kernelId", "name", "MPerBLOCK", "NPerBLOCK", "KPerBLOCK",
                    "MFMA_M", "MFMA_N", "WAVE_MAP_M", "WAVE_MAP_N",
                    "BLOCK_SIZE", "pipeline_version"])
        for kid in sorted(table):
            ki = table[kid]
            w.writerow([kid, ki.name, *ki.mt, *ki.mi_mn, *ki.waves,
                        ki.block_size, ki.pipeline_version])
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*", default=list(FAMILIES))
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "results"))
    args = ap.parse_args()
    for key in args.family:
        fam = get_family(key)
        try:
            table = load_kernel_table(fam)
            path = dump_table(fam, args.out_dir)
            print(f"  [{key}] {len(table)} kernelIds -> {os.path.relpath(path, _ROOT)}")
        except Exception as e:
            print(f"  [{key}] FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
