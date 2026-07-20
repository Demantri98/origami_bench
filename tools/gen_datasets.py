#!/usr/bin/env python3
"""Generate novel (M, N, K) datasets for the CK dense GEMM families.

Each dataset holds ~N points on gfx950 that DO NOT appear in that family's
tuned config CSV(s) (main + model_configs), so every point exercises aiter's
missing-shape fallback (heuristic / default CK instance). Shapes stay realistic
and runnable:

  * (N, K) come from real weight shapes seen for the family, PLUS synthesized
    cross-combinations (real N x real K) that respect the family's divisibility
    constraints. No (N, K) is invented from thin air.
  * M is drawn from a realistic, decode->prefill weighted distribution with both
    "round" and deliberately "odd" values so many tuples are novel.

Runs anywhere (pure python, no torch / GPU). Deterministic given --seed.

Usage:
  python3 tools/gen_datasets.py                 # all families, 1000 each
  python3 tools/gen_datasets.py --n 1000 --seed 0 --families a8w8 a4w4_blockscale
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.families import FAMILIES, TARGET_GFX, Family, get_family  # noqa: E402
from common.shapes import load_existing, load_nk_pool  # noqa: E402

# Cap dims to a realistic dense-GEMM range. A few pool (N,K) values (e.g. K in
# the 200k-600k range) are flattened MoE/fused dims that leaked into a8w8
# model_configs; they OOM or are unsupported on the dense CK path, so exclude
# them to keep every dataset point runnable.
MAX_DIM = 65536

# Realistic M tiers: (values, relative weight). Weighted toward the decode /
# small-batch regime that dominates LLM inference, with a long prefill tail.
_M_TIERS: list[tuple[list[int], float]] = [
    (list(range(1, 17)), 3.0),                                   # decode
    ([17, 20, 24, 28, 32, 40, 48, 56, 64], 2.5),                 # small batch
    ([80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512], 2.0),
    ([640, 768, 896, 1024, 1280, 1536, 2048, 2560, 3072, 4096], 1.5),  # medium / prefill
    ([5120, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536], 0.8),
    # deliberately "odd" values -> more likely to be novel vs tuned round shapes
    ([3, 5, 7, 9, 11, 13, 33, 100, 200, 300, 333, 500, 777,
      1000, 1234, 1500, 2000, 3000, 3333, 6000, 10000, 20000, 40000], 1.2),
]


def _weighted_m_choices(rng: random.Random, count: int) -> list[int]:
    """Sample ``count`` M values (with replacement) per the tier weights."""
    tiers, weights = zip(*_M_TIERS)
    picked = []
    for _ in range(count):
        tier = rng.choices(tiers, weights=weights, k=1)[0]
        picked.append(rng.choice(tier))
    return picked


def _synthesize_nk(fam: Family, pool: list[tuple[int, int]],
                   rng: random.Random, extra: int) -> list[tuple[int, int]]:
    """Real (N, K) pool + up to ``extra`` novel cross-combos (real N x real K).

    Cross-combos preserve the family's divisibility constraints (they come from
    real Ns/Ks that already satisfy them), so they remain runnable.
    """
    ns = sorted({n for n, _ in pool})
    ks = sorted({k for _, k in pool})
    have = set(pool)
    combos: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    # bounded attempts to avoid pathological loops on tiny pools
    attempts = 0
    max_attempts = extra * 50 + 1000
    while len(combos) < extra and attempts < max_attempts:
        attempts += 1
        n = rng.choice(ns)
        k = rng.choice(ks)
        if n % fam.n_mod or k % fam.k_mod:
            continue
        if n < fam.min_n or k < fam.min_k:
            continue
        cand = (n, k)
        if cand in have or cand in seen:
            continue
        seen.add(cand)
        combos.append(cand)
    return pool + combos


def _pick_dtype(fam: Family, rng: random.Random) -> str:
    toks, weights = zip(*fam.dtype_weights.items())
    return rng.choices(list(toks), weights=list(weights), k=1)[0]


def generate(fam: Family, n_points: int, seed: int) -> list[tuple]:
    rng = random.Random(f"{fam.key}:{seed}")
    existing = load_existing(fam, TARGET_GFX)
    pool = [nk for nk in load_nk_pool(fam, TARGET_GFX)
            if fam.min_n <= nk[0] <= MAX_DIM and fam.min_k <= nk[1] <= MAX_DIM]
    if not pool:
        raise RuntimeError(f"empty (N,K) pool for {fam.key}")

    # ~40% of the (N,K) variety comes from synthesized cross-combos.
    nk_candidates = _synthesize_nk(fam, pool, rng, extra=max(len(pool) // 2, 64))

    rows: list[tuple] = []
    chosen: set[tuple] = set()
    # Draw M values in bulk, refill as needed.
    m_batch: list[int] = []
    m_idx = 0
    guard = 0
    guard_max = n_points * 400 + 20000
    while len(rows) < n_points and guard < guard_max:
        guard += 1
        if m_idx >= len(m_batch):
            m_batch = _weighted_m_choices(rng, max(n_points, 2000))
            m_idx = 0
        m = m_batch[m_idx]
        m_idx += 1
        if m < fam.min_m:
            continue
        n, k = rng.choice(nk_candidates)
        if fam.has_dtype:
            dt = _pick_dtype(fam, rng)
            key = (m, n, k, dt)
        else:
            key = (m, n, k)
        if key in existing or key in chosen:
            continue
        chosen.add(key)
        rows.append(key)

    if len(rows) < n_points:
        raise RuntimeError(
            f"{fam.key}: only produced {len(rows)}/{n_points} novel points "
            f"(pool exhausted). Increase M tiers or cross-combos."
        )
    # Sort for stable, human-readable output (by N, K, M[, dtype]).
    rows.sort(key=lambda r: (r[1], r[2], r[0]) + (r[3:] if len(r) > 3 else ()))
    return rows


def write_csv(fam: Family, rows: list[tuple], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = fam.dataset_path(out_dir)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fam.schema)
        for r in rows:
            w.writerow(r)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="points per family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "datasets"))
    ap.add_argument("--families", nargs="*", default=list(FAMILIES),
                    help="subset of families to generate")
    args = ap.parse_args()

    print(f"Generating novel gfx950 datasets ({args.n} points each, seed={args.seed})\n")
    for key in args.families:
        fam = get_family(key)
        rows = generate(fam, args.n, args.seed)
        path = write_csv(fam, rows, args.out_dir)
        # sanity: none of the generated points may collide with existing tuned shapes
        existing = load_existing(fam, TARGET_GFX)
        overlap = sum(1 for r in rows if r in existing)
        n_distinct_nk = len({(r[1], r[2]) for r in rows})
        m_lo = min(r[0] for r in rows)
        m_hi = max(r[0] for r in rows)
        assert overlap == 0, f"{key}: {overlap} generated points overlap existing!"
        print(f"  {key:30s} {len(rows):5d} rows  "
              f"({n_distinct_nk} distinct (N,K), M in [{m_lo},{m_hi}])  -> {os.path.relpath(path, _ROOT)}")
    print("\nDone. All datasets verified disjoint from tuned CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
