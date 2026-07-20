# Origami-picked CK kernel vs aiter default fallback — validation

**Question:** for GEMM shapes with no tuned entry on gfx950 (where aiter runs a
built-in default/heuristic CK instance), does letting **Origami** pick the CK
kernel as a backup heuristic produce a *faster and correct* kernel than aiter's
default?

## Setup

- **Hardware:** gfx950 (MI355X, 256 CU), container `origami-dev`.
- **Shapes:** random 150-shape sample per family (`--sample 150 --sample-seed 1`)
  drawn from the novel-shape datasets (each point absent from aiter's tuned CSVs).
- **Selection:** Origami ranks one `config_t` per real aiter `kernelId` (the
  `ck_kernel_map` linchpin) via `compute_total_latency`; argmin = the pick.
- **splitK = 0** for the Origami pick (Origami does not model splitK, and aiter's
  default fallback also uses splitK=0 on a miss — so this isolates *kernel choice*).
- **Timing:** hipEvents, 3 warmup + 30 iters, device time per call.
- **Correctness gate:** the Origami-picked kernel's output must match aiter's
  trusted production output on identical inputs (`torch.isclose` rtol=atol=1e-2,
  ≤2% mismatched elements). Picks that fail are **excluded** from win-rate.

## Results

| family | correct/measured | correctness fails | Origami win-rate | speedup geomean | median | mean TFLOPS default → origami |
|---|---|---|---|---|---|---|
| **a8w8_blockscale** | 93 | 0 | **90%** | **1.71×** | 1.60× | 212 → **630** |
| **a8w8_blockscale_bpreshuffle** | 80 | 0 | **71%** | 1.03× | 1.21× | 330 → **532** |
| a8w8 (rowwise) | 136 | 2 | 49% | 0.98× | 1.00× | 411 → 356 |
| a8w8_bpreshuffle | 133 | 0 | 45% | **0.83×** | 0.94× | 493 → 428 |
| a4w4_blockscale | — | — | — | — | — | **see below** |

Win-rate by M regime:

| family | small-M (≤128) | large-M (>128) |
|---|---|---|
| a8w8_blockscale | 91% (n=22) | 90% (n=71) |
| a8w8_blockscale_bpreshuffle | 100% (n=20) | 62% (n=60) |
| a8w8 | 54% (n=84) | 40% (n=52) |
| a8w8_bpreshuffle | 37% (n=84) | 59% (n=49) |

*(speedup >1 = Origami's pick faster; geomean/median of per-shape ratios — the
arithmetic mean of ratios is biased and not used.)*

## Findings

1. **Origami is a clear, validated win for `a8w8_blockscale`: ~1.7× geomean,
   90% win-rate, 0 correctness failures, across all M.** This is mechanistic:
   aiter's blockscale fallback is a *single fixed default kernel* (not
   shape-aware), so a shape-aware pick wins almost everywhere. Mean throughput
   nearly triples (212 → 630 TFLOPS).

2. **`a8w8_blockscale_bpreshuffle` is a moderate win** (71% win, median 1.21×,
   strongest at small M). Same mechanism, weaker margin.

3. **`a8w8` rowwise and `a8w8_bpreshuffle` are neutral-to-worse** (geomean 0.98
   and 0.83). These families already have a *shape-aware C++ heuristic tree* as
   their fallback, so Origami adds little and for bpreshuffle is a net loss.

4. **`a4w4_blockscale` picks are unsafe.** Even at splitK=0, some Origami-picked
   a4w4 kernels produce numerically wrong output (observed err ratio 0.80), and
   one pick **deterministically triggers a GPU fault (core dump)** that aborts the
   process. a4w4 is excluded pending per-shape process isolation + tighter kernel
   feasibility checks. This is itself a result: Origami's analytical model does
   not know which a4w4 CK instances are valid for a given shape.

## Recommendation

Wire Origami in as the backup heuristic **specifically for the block-scaled
families** (`a8w8_blockscale`, and secondarily `a8w8_blockscale_bpreshuffle`),
where aiter's fallback is a fixed default and the win is large and correct. **Do
not** use it for `a8w8`/`a8w8_bpreshuffle` (already have a good heuristic; net
neutral/negative) or `a4w4` (unsafe picks) without further work.

## Key correctness lesson (why the gate mattered)

An earlier run *without* splitK=0 credited Origami with big a8w8 (1.22×) and a4w4
wins — but the correctness gate showed **100% of those splitK>0 runs were
numerically wrong** (an ad-hoc splitK heuristic, not part of Origami, corrupted
the output). Isolating to splitK=0 removed 31/33 a8w8 "failures". Without the
gate, the headline numbers would have been materially overstated.

## Caveats

- Ranking granularity is `(macrotile, MFMA)`; occupancy/pipeline variants collapse
  to one `config_t`, so tied kernelIds are indistinguishable to Origami.
- Origami declines (returns infeasible) ~37% of blockscale shapes; those fall back
  to the default and are not in the win-rate.
- hipEvent timing at 30 iters has some noise on tiny shapes.
- Correctness is measured against aiter's production output (a trusted reference),
  not an independent fp32 recomputation.

## Reproduce

```bash
docker exec origami-dev bash -lc 'cd /home/demantri/ck_origami_bench && \
  python3 tools/origami_select.py --candidates aiter --sample 150 --sample-seed 1 && \
  python3 tools/aiter_run.py --mode prod --sample 150 --sample-seed 1 --iters 30 --warmup 3 && \
  python3 tools/compare.py'
```

Per-shape data: `results/<family>_compare.csv`; aggregate: `results/summary.csv`.
