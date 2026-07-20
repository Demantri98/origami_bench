"""Registry of the 5 CK *dense* 2D GEMM families in aiter.

This module is intentionally free of ``torch`` / ``aiter`` imports so it can be
imported from any environment: the dataset generator (host py3.9), the Origami
selection tool (py3.12 origami container), and the aiter benchmark tool (host
py3.9 + GPU) all share the same definitions.

Every family keys its tuned config on ``(gfx, cu_num, M, N, K[, q_dtype_w])`` and
falls back to a C++ heuristic / default CK instance when a shape is missing (see
``aiter/aiter/ops/gemm_op_a8w8.py`` and ``gemm_op_a4w4.py``). The datasets we
generate are exactly the *missing* ``(M, N, K)`` points on gfx950, so they
exercise that fallback path — the regime where Origami's analytical ranking is
meant to help.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Repo layout ---------------------------------------------------------------
AITER_ROOT = os.environ.get("AITER_ROOT", "/home/demantri/aiter")
CONFIG_DIR = os.path.join(AITER_ROOT, "aiter", "configs")
MODEL_CONFIG_DIR = os.path.join(CONFIG_DIR, "model_configs")

# Target arch for every dataset. gfx950 (MI355X) is what this box runs.
TARGET_GFX = "gfx950"

# dtype string tokens for aiter's ``q_dtype_w`` column.
DT_INT8 = "torch.int8"
DT_FP8 = "torch.float8_e4m3fn"  # gfx950 fp8 == OCP e4m3fn (see aiter/utility/dtypes.py)


@dataclass(frozen=True)
class Family:
    """Static description of one CK dense GEMM family."""

    key: str
    # Main tuned CSV aiter loads at runtime (relative to CONFIG_DIR).
    tuned_csv: str
    # Substrings identifying this family's model_configs CSVs. A model_configs
    # file matches when ALL of the family's ``mc_include`` tokens appear in the
    # name AND none of the ``mc_exclude`` tokens do. This disambiguates e.g.
    # "a8w8_blockscale" vs "a8w8_blockscale_bpreshuffle" vs plain "a8w8".
    mc_include: tuple[str, ...]
    mc_exclude: tuple[str, ...]
    # Dataset CSV columns.
    schema: tuple[str, ...]
    # Whether rows carry a per-row weight quant dtype (a8w8 / bpreshuffle).
    has_dtype: bool
    # Allowed q_dtype_w tokens with sampling weights (only if has_dtype).
    dtype_weights: dict[str, float]
    # Divisibility constraints kept when synthesizing novel (N, K) combos so the
    # shapes stay runnable (weight preshuffle / block-scale alignment).
    n_mod: int
    k_mod: int
    # Origami problem_t dtype mapping (a/b operand, output, matrix-instruction),
    # keyed by q_dtype_w token; use key "" for families without a dtype column.
    origami_dtypes: dict[str, dict]
    # bytes-per-element (in, weight, out) for TFLOPS/bandwidth accounting.
    bpes: tuple[float, float, float]
    # --- CK kernel instance table (the "linchpin" mapping source) ---------- #
    # csrc subdir + module + dict attribute holding {kernelId: kernelInstance}.
    # These common modules are pure-python dataclasses (no torch), so the tables
    # load without a GPU. Each instance carries MPerBLOCK/NPerBLOCK/KPerBLOCK
    # (macrotile) and an MFMA size in either WAVE_TILE_{M,N} or MPerXDL/NPerXDL.
    csrc_subdir: str
    kernel_module: str
    kernel_dict_attr: str
    # Minimum runnable dimensions on the CK path. Some tiny shapes are ASM-only
    # (e.g. blockscale_bpreshuffle rejects N<64 with "This GEMM is not
    # supported!"); such points are excluded so datasets stay runnable on CK.
    min_m: int = 1
    min_n: int = 1
    min_k: int = 1
    # Human note.
    note: str = ""

    def dataset_path(self, datasets_dir: str) -> str:
        return os.path.join(datasets_dir, f"{self.key}_gfx950_novel.csv")


# Origami dtype presets (token strings passed to origami.string_to_datatype;
# the tool degrades gracefully if a token is unknown to this origami build).
_OD_INT8 = {"a": "i8", "b": "i8", "out": "bf16", "mi": "i8", "mx": 0}
_OD_FP8 = {"a": "f8", "b": "f8", "out": "bf16", "mi": "f8", "mx": 0}
_OD_FP4 = {"a": "f4", "b": "f4", "out": "bf16", "mi": "f4", "mx": 32}


FAMILIES: dict[str, Family] = {
    "a8w8": Family(
        key="a8w8",
        tuned_csv="a8w8_tuned_gemm.csv",
        mc_include=("a8w8", "tuned_gemm"),
        mc_exclude=("blockscale", "bpreshuffle", "batched", "fmoe", "bf16", "a4w4"),
        schema=("M", "N", "K", "q_dtype_w"),
        has_dtype=True,
        # gfx950 tuned rows are ~82% int8 / ~18% fp8; mirror that split.
        dtype_weights={DT_INT8: 0.7, DT_FP8: 0.3},
        n_mod=16,
        k_mod=16,
        origami_dtypes={DT_INT8: _OD_INT8, DT_FP8: _OD_FP8},
        bpes=(1, 1, 2),
        csrc_subdir="ck_gemm_a8w8",
        kernel_module="gemm_a8w8_common",
        kernel_dict_attr="kernels_list",
        note="Rowwise per-token scaled int8/fp8 GEMM. Weight [N,K] row-major.",
    ),
    "a8w8_bpreshuffle": Family(
        key="a8w8_bpreshuffle",
        tuned_csv="a8w8_bpreshuffle_tuned_gemm.csv",
        mc_include=("a8w8_bpreshuffle", "tuned_gemm"),
        mc_exclude=("blockscale", "batched", "fmoe"),
        schema=("M", "N", "K", "q_dtype_w"),
        has_dtype=True,
        # gfx950 CK/asm bpreshuffle rows are fp8 only.
        dtype_weights={DT_FP8: 1.0},
        n_mod=16,
        k_mod=32,  # shuffle_weight(layout=(16,16)) on fp8 -> BK=32
        origami_dtypes={DT_FP8: _OD_FP8},
        bpes=(1, 1, 2),
        csrc_subdir="ck_gemm_a8w8_bpreshuffle",
        kernel_module="gemm_a8w8_bpreshuffle_common",
        kernel_dict_attr="kernels_list",
        note="Rowwise fp8 GEMM with preshuffled weights (shuffle_weight 16x16).",
    ),
    "a8w8_blockscale": Family(
        key="a8w8_blockscale",
        tuned_csv="a8w8_blockscale_tuned_gemm.csv",
        mc_include=("a8w8_blockscale", "tuned_gemm"),
        mc_exclude=("bpreshuffle", "batched", "fmoe"),
        schema=("M", "N", "K"),
        has_dtype=False,
        dtype_weights={},
        n_mod=16,
        k_mod=128,  # 1x128x128 block scaling
        origami_dtypes={"": _OD_FP8},
        bpes=(1, 1, 2),
        csrc_subdir="ck_gemm_a8w8_blockscale",
        kernel_module="gemm_a8w8_blockscale_instance",
        kernel_dict_attr="candidate_kernels_dict",
        note="fp8 GEMM, 1x128x128 block scales. Weight [N,K] row-major.",
    ),
    "a8w8_blockscale_bpreshuffle": Family(
        key="a8w8_blockscale_bpreshuffle",
        tuned_csv="a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        mc_include=("a8w8_blockscale_bpreshuffle", "tuned_gemm"),
        mc_exclude=("batched", "fmoe"),
        schema=("M", "N", "K"),
        has_dtype=False,
        dtype_weights={},
        n_mod=16,
        k_mod=128,
        origami_dtypes={"": _OD_FP8},
        bpes=(1, 1, 2),
        csrc_subdir="ck_gemm_a8w8_blockscale_bpreshuffle",
        kernel_module="gemm_a8w8_blockscale_bpreshuffle_common",
        kernel_dict_attr="kernels_list",
        min_n=64,  # CK path rejects N<64 (16/32 are ASM-only on gfx950)
        note="fp8 block-scaled GEMM with preshuffled weights + transposed x_scale.",
    ),
    "a4w4_blockscale": Family(
        key="a4w4_blockscale",
        tuned_csv="a4w4_blockscale_tuned_gemm.csv",
        mc_include=("a4w4", "tuned_gemm"),
        mc_exclude=("fmoe", "batched"),
        schema=("M", "N", "K"),
        has_dtype=False,
        dtype_weights={},
        n_mod=16,
        k_mod=64,  # fp4x2 pack + shuffle_weight -> (K/2)%32==0
        origami_dtypes={"": _OD_FP4},
        bpes=(0.5, 0.5, 2),
        csrc_subdir="ck_gemm_a4w4_blockscale",
        kernel_module="gemm_a4w4_blockscale_common",
        kernel_dict_attr="kernels_list",
        note="MXFP4 (fp4x2) GEMM, e8m0 block scales (block=32), preshuffled weight.",
    ),
}


def get_family(key: str) -> Family:
    if key not in FAMILIES:
        raise KeyError(f"unknown family '{key}'. known: {sorted(FAMILIES)}")
    return FAMILIES[key]
