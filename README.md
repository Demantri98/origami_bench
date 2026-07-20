# CK Dense GEMM × Origami benchmark harness

Datasets and tooling to evaluate **Origami** (analytical CK config ranker) against
**aiter**'s real CK dense-GEMM dispatch on **gfx950**, over problem shapes that
have **no tuned entry** in aiter's config CSVs (i.e. the fallback regime, where a
missing `(gfx, cu_num, M, N, K)` resolves to a C++ heuristic / default CK
instance — exactly what Origami is meant to improve).

**Core idea:** everything runs in **prod** (single-kernel, no exhaustive search).
For an untuned shape, aiter today runs a built-in default/heuristic CK instance.
Origami acts as a **backup heuristic**: it selects a concrete aiter CK `kernelId`
analytically, we run *that* kernel, and compare it against the default. The
`kernelId`↔`config_t` mapping that makes this possible (the "linchpin") lives in
`tools/ck_kernel_map.py`.

## Families covered (CK dense 2D GEMM)

| key | aiter op | dtypes | tuned CSV keyed on |
|---|---|---|---|
| `a8w8` | `gemm_a8w8` | int8 / fp8 rowwise | `M,N,K,q_dtype_w` |
| `a8w8_bpreshuffle` | `gemm_a8w8_bpreshuffle` | fp8, preshuffled W | `M,N,K,q_dtype_w` |
| `a8w8_blockscale` | `gemm_a8w8_blockscale` | fp8, 1×128×128 scales | `M,N,K` |
| `a8w8_blockscale_bpreshuffle` | `gemm_a8w8_blockscale_bpreshuffle` | fp8, blockscale + preshuffle | `M,N,K` |
| `a4w4_blockscale` | `gemm_a4w4_blockscale` | mxfp4 (fp4x2), e8m0 block=32 | `M,N,K` |

(Batched, grouped/DeepGEMM and MoE GEMMs are intentionally out of scope — this is
"dense" 2D GEMM only.)

## Layout

```
ck_origami_bench/
├── common/
│   ├── families.py   # single source of truth: family registry (torch-free)
│   └── shapes.py     # load existing tuned shapes + realistic (N,K) pools
├── datasets/         # generated: <family>_gfx950_novel.csv  (~1000 rows each)
├── results/          # tool outputs (origami / aiter / compare CSVs)
└── tools/
    ├── gen_datasets.py    # build the novel-shape datasets (no GPU)
    ├── ck_kernel_map.py   # LINCHPIN: aiter kernelId <-> Origami config_t
    ├── origami_select.py  # pick a concrete aiter kernelId per shape via Origami
    ├── aiter_run.py       # prod-run + benchmark (default + Origami-picked) (GPU)
    └── compare.py         # summarize default-vs-Origami win rate / speedup
```

## The linchpin: kernelId ↔ config_t

Each family ships a **pure-python** table of CK kernel instances
(`csrc/<fam>/*_common*.py`, keyed by `kernelId`), each carrying a macrotile
`(MPerBLOCK, NPerBLOCK, KPerBLOCK)`, an MFMA size (`WAVE_TILE_{M,N}` or
`MPerXDL/NPerXDL`) and a wave mapping. `tools/ck_kernel_map.py` builds **one
Origami `config_t` per real `kernelId`** (`mt`=macrotile, `mi`=MFMA, constant
`occupancy`) and scores them with `compute_total_latency`. The argmin is
therefore *both* Origami's recommended config *and* a concrete aiter CK instance
— no fuzzy tile matching. Dump the tables (no GPU needed):

```bash
python3 tools/ck_kernel_map.py            # -> results/<family>_ck_kernels.csv
```

## Dataset semantics

Each `datasets/<family>_gfx950_novel.csv` holds ~1000 `(M, N, K[, q_dtype_w])`
points that are **disjoint** from every gfx950 row in that family's tuned CSV
(`aiter/aiter/configs/<family>_tuned_gemm.csv`) **and** its `model_configs/`
overrides. Shapes stay realistic and runnable:

- `(N, K)` are real weight shapes seen for the family, plus synthesized
  cross-combinations (real N × real K) that preserve the family's divisibility
  constraints. No dimension is invented from thin air.
- `M` follows a decode→prefill weighted distribution (≈52% of points have M≤64)
  with deliberately "odd" values mixed in to guarantee novelty.
- Points below the CK path's minimum runnable size are excluded (e.g.
  `a8w8_blockscale_bpreshuffle` needs `N≥64`; N=16/32 are ASM-only).

Regenerate deterministically:

```bash
python3 tools/gen_datasets.py --n 1000 --seed 0        # all families
python3 tools/gen_datasets.py --families a4w4_blockscale --n 500
```

## Environment (important)

- The **host** Python 3.9 has neither `torch` nor a real `aiter` install.
- Both `aiter` (with prebuilt CK `.so`s + GPU) **and** `origami` (py3.12 nanobind
  wheel) are available inside the **`origami-dev`** container
  (`vllm/vllm-openai-rocm:v0.19.0`), which mounts `/home/demantri`. Run all
  GPU/Origami tools there:

```bash
docker start origami-dev
docker exec origami-dev bash -lc 'cd /home/demantri/ck_origami_bench && <command>'
```

`gen_datasets.py` and `compare.py` are pure-python and also run on the host.

## Usage (the prod flow)

Run all three inside `origami-dev`. The order matters: Origami selects the
kernelId first, then aiter runs both the default and the Origami-picked kernel.

**1. Origami picks a concrete kernelId per shape** (reads the live gfx950 device):

```bash
docker exec origami-dev bash -lc 'cd /home/demantri/ck_origami_bench && \
  python3 tools/origami_select.py --family a8w8_blockscale --candidates aiter'
# -> results/a8w8_blockscale_origami.csv  (best_kernelId, best_kernelName, splitK, ...)
```

`--candidates aiter` (default) ranks the real aiter kernelIds (the linchpin).
`--candidates grid` ranks an abstract tile grid instead (no aiter needed; emits
tiles only). Off a ROCm/origami box it writes an `available=false` marker and
exits 0 (safe dry-run on the host).

**2. aiter prod run** — measures the native default fallback **and** (if the
Origami picks from step 1 are present) the Origami-selected kernel, per shape:

```bash
docker exec origami-dev bash -lc 'cd /home/demantri/ck_origami_bench && \
  python3 tools/aiter_run.py --family a8w8_blockscale --mode prod'
# -> results/a8w8_blockscale_aiter_prod.csv
#    columns: prod_* (default fallback), origami_* (Origami-picked kernel),
#             winner, origami_speedup_vs_prod
```

Add `--no-origami` to measure only the native fallback. The optional
`--mode sweep` / `--mode both` runs the exhaustive oracle (all kernelIds × splitK)
if you want to know how near-optimal each pick is — slow, use `--limit`.

**3. Summarize:**

```bash
python3 tools/compare.py --family a8w8_blockscale
# -> results/<family>_compare.csv + win-rate / mean-speedup / (frac-of-oracle) summary
```

## Notes / caveats

- **Occupancy** is not encoded in the CK instance table (it is a compiler/launch
  property), so `ck_kernel_map` holds it constant across instances — Origami
  therefore ranks at `(macrotile, MFMA)` granularity. Multiple kernelIds that
  differ only in pipeline/transfer params collapse to the same `config_t`.
- Predicted latency is in **GPU cycles** — a ranking signal, not a wall time.
  Compare relative ordering, not absolute magnitude, against measured µs.
  Infeasible tiles (Origami returns a ~DBL_MAX sentinel) are filtered before the
  argmin.
- An Origami-picked kernelId can still fail to run for a given shape (recorded as
  `origami_status=error`); those rows simply have no `origami_us`.
- `--mode sweep`/`both` compiles `module_gemm_*_tune` on first use (tens of
  seconds) and is O(#kernels × #splitK) per shape.
