#!/usr/bin/env python3
"""Mechanistic, workload-driven synthetic (M, N, K) generator (no GPU, no training).

Unlike ``gen_datasets.py`` (which samples M from hand-written tiers and mixes in
synthesized cross-combinations of (N, K)), this tool treats M as what it
physically *is* for a serving GEMM: the number of rows in the activation matrix
fed to one kernel invocation, i.e.

    M  =  prefill_tokens_this_step  +  concurrent_decode_sequences

and builds it from an interpretable mini serving model instead of an opaque
density fit. Every knob means something ("what if avg prompt length = 2k and
concurrency = 32?"), and the defaults are *calibrated to a real tuned CSV* so the
generated M distribution resembles the observed one per (N, K).

Design (per generated GEMM = one scheduler step):
  * With probability ``p_decode`` the step is decode-only  ->  M = c
    (c = number of sequences currently generating tokens; the small-M regime).
  * Otherwise it is a chunked-prefill step  ->  M = min(prompt, B - c) + c
    where ``prompt`` is a request prompt length drawn from a K-mode LogNormal
    mixture over context lengths (short / medium / long-context), ``B`` =
    ``max_num_batched_tokens`` is the per-step token budget, and ``c`` is the
    co-running decode concurrency.
  * ``c`` ~ round( Beta(conc_a, conc_b) * max_num_seqs ), clamped to [1, S].
    One concurrency distribution therefore explains BOTH the decode regime and
    the small prefill offset (mechanistically consistent).

The prompt-length mixture is calibrated to the real CSV by EM on the per-row
implied prompt lengths (M - E[concurrency]) over the prefill regime; the budget
cap ``B`` then governs the very top of the distribution at generation time.

(N, K) are sampled ONLY from the real weight shapes in the family's tuned CSV,
by their observed frequency (no invented / cross-combined shapes).

Every emitted (M, N, K) is guaranteed disjoint from the family's tuned CSV(s)
(main + model_configs), and unique within the output.

Runs anywhere (numpy only; no torch / aiter / GPU). Deterministic given --seed.

Usage:
  # calibrate to the real CSV, print the fitted knobs, generate ~same #rows:
  python3 tools/gen_workload_dataset.py --family a8w8_blockscale

  # "what if" exploration (shape stays calibrated; these knobs shift it):
  python3 tools/gen_workload_dataset.py --family a8w8_blockscale \
      --prompt-scale 2.0 --max-num-seqs 32 --p-decode 0.5 --n 2000

  # stratified-by-M-regime mode: for an explicit set of (N,K) weight shapes,
  # emit a fixed number of M values in EACH serving regime (decode / small-batch
  # / medium-prefill / large-prefill). Every emitted (M,N,K) is guaranteed to
  # miss the tuned CSV under aiter's runtime M-padding, i.e. to dispatch to the
  # family's DEFAULT CK kernel on the target (gfx, cu_num):
  python3 tools/gen_workload_dataset.py --family a8w8_blockscale --stratified \
      --nk "51200,5120;5120,25600;10240,5120;5120,8192" --per-regime 12
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.families import TARGET_GFX, get_family  # noqa: E402
from common.shapes import family_csv_files, load_existing, load_nk_pool  # noqa: E402

# Boundary (tokens) between the decode / small-batch regime and the prefill
# regime, used only to *fit* knobs from the real data. Chosen at the elbow of the
# observed M histogram (the hand-tuned decode grid tops out at 512).
_DECODE_MAX = 512


@dataclass
class Workload:
    """Interpretable serving-workload knobs (the whole model).

    Request prompt length is a K-mode LogNormal mixture over context lengths
    (e.g. short/interactive, medium, and long/near-cap RAG requests). This
    multi-modal context-length structure is what gives the real M its fat lower
    body, its dominant mid mode, AND its heavy upper shoulder near the token
    budget.
    """

    p_decode: float          # fraction of steps that are decode-only
    # prompt length mixture (tokens): parallel lists, one entry per context mode
    prompt_w: list            # mixture weights (sum to 1)
    prompt_mu: list           # LogNormal mu per mode
    prompt_sigma: list        # LogNormal sigma per mode
    prompt_scale: float       # global multiplier on prompt length ("what if 2x?")
    max_num_batched_tokens: int   # per-step token budget B (chunked-prefill cap)
    max_num_seqs: int        # max concurrent sequences S (concurrency cap)
    conc_a: float            # Beta(a,b) shape for concurrency c = round(Beta*S)
    conc_b: float

    def concurrency_mean(self) -> float:
        return self.max_num_seqs * self.conc_a / (self.conc_a + self.conc_b)

    def mode_medians(self) -> list:
        return [math.exp(mu) * self.prompt_scale for mu in self.prompt_mu]


def _read_real_mnk(fam, gfx: str):
    """Real (M, N, K) rows for ``gfx`` from the family's main tuned CSV."""
    import pandas as pd

    main = os.path.join(os.path.dirname(family_csv_files(fam)[0]), fam.tuned_csv)
    df = pd.read_csv(main)
    if "gfx" in df.columns:
        df = df[df["gfx"] == gfx]
    return df[["M", "N", "K"]].astype(int)


def fit_workload(fam, gfx: str = TARGET_GFX,
                 conc_a: float = 2.0, conc_b: float = 3.0,
                 n_modes: int = 4) -> Workload:
    """Calibrate workload knobs from the real tuned CSV.

    * p_decode           = observed fraction of rows with M <= _DECODE_MAX
    * max_num_batched_tokens = observed max M (the token budget / cap)
    * max_num_seqs       = concurrency scale chosen so E[c] matches the mean of
      the decode-regime M (which physically *is* the concurrency).
    * prompt-length mixture = fit by EM (see ``_calibrate_prompt``).
    """
    mnk = _read_real_mnk(fam, gfx)
    M = mnk["M"].to_numpy()
    decode = M[M <= _DECODE_MAX]
    prefill = M[M > _DECODE_MAX]
    if prefill.size == 0:
        raise RuntimeError(f"{fam.key}: no prefill-regime rows to calibrate from")

    p_decode = float((M <= _DECODE_MAX).mean())
    cap = int(M.max())

    # Concurrency: mean decode-regime M == mean concurrency E[c] = S*a/(a+b).
    conc_mean = float(decode.mean()) if decode.size else 0.4 * _DECODE_MAX
    max_num_seqs = max(1, int(round(conc_mean * (conc_a + conc_b) / conc_a)))

    wl = Workload(
        p_decode=p_decode,
        prompt_w=[1.0],
        prompt_mu=[float(math.log(np.median(prefill)))],
        prompt_sigma=[float(np.log(prefill).std())],
        prompt_scale=1.0,
        max_num_batched_tokens=cap,
        max_num_seqs=max_num_seqs,
        conc_a=conc_a,
        conc_b=conc_b,
    )
    # Calibrate the prompt-length mixture to the CSV (EM on implied prompt
    # lengths). Same mechanistic model; only the context-length modes are fit.
    _calibrate_prompt(wl, M, n_modes=n_modes)
    return wl


def _em_lognormal_mixture(x: np.ndarray, k: int, iters: int = 500,
                          min_sigma: float = 0.12):
    """Fit a K-component 1-D Gaussian mixture to ``x`` (values in log space).

    Returns (w, mu, sigma) arrays sorted by mu ascending. Pure numpy EM,
    deterministic (means initialized at evenly-spaced quantiles), sub-second.
    ``min_sigma`` floors component widths to avoid degenerate near-delta modes
    (which would generate many identical rows).
    """
    x = x.astype(float)
    qs = np.linspace(0.5 / k, 1 - 0.5 / k, k)
    mu = np.quantile(x, qs)
    sig = np.full(k, max(x.std() / k, min_sigma))
    w = np.full(k, 1.0 / k)
    for _ in range(iters):
        d = x[:, None] - mu[None, :]
        logp = -0.5 * (d / sig[None, :]) ** 2 - np.log(sig[None, :]) + np.log(w[None, :])
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-9
        w_new = nk / nk.sum()
        mu_new = (r * x[:, None]).sum(axis=0) / nk
        var = (r * (x[:, None] - mu_new[None, :]) ** 2).sum(axis=0) / nk
        sig_new = np.maximum(np.sqrt(np.maximum(var, 1e-6)), min_sigma)
        if np.allclose(mu_new, mu, atol=1e-7) and np.allclose(w_new, w, atol=1e-7):
            mu, sig, w = mu_new, sig_new, w_new
            break
        mu, sig, w = mu_new, sig_new, w_new
    order = np.argsort(mu)
    return w[order], mu[order], sig[order]


def _calibrate_prompt(wl: Workload, real_M: np.ndarray, n_modes: int = 4) -> None:
    """Calibrate the prompt-length mixture to the real M via EM.

    We recover per-row *implied prompt lengths* on the prefill regime
    (prompt ≈ M − E[concurrency]) and fit a K-component log-normal mixture to
    them in log space. The distinct near-cap (long-context) cluster becomes its
    own mode, which — combined with the budget clip at generation — reproduces
    the observed upper shoulder. No simulation in the loop.
    """
    conc_mean = wl.concurrency_mean()
    prefill = real_M[real_M > _DECODE_MAX].astype(float)
    implied = np.clip(prefill - conc_mean, 1.0, None)
    w, mu, sig = _em_lognormal_mixture(np.log(implied), k=n_modes)
    wl.prompt_w = [float(v) for v in w]
    wl.prompt_mu = [float(v) for v in mu]
    wl.prompt_sigma = [float(v) for v in sig]


def _sample_concurrency(rng: np.random.Generator, wl: Workload, size: int) -> np.ndarray:
    c = np.round(rng.beta(wl.conc_a, wl.conc_b, size=size) * wl.max_num_seqs)
    return np.clip(c, 1, wl.max_num_seqs).astype(np.int64)


def _sample_prompt(rng: np.random.Generator, wl: Workload, size: int) -> np.ndarray:
    """K-mode LogNormal mixture of request prompt lengths (tokens)."""
    w = np.asarray(wl.prompt_w, dtype=float)
    w = w / w.sum()
    comp = rng.choice(len(w), size=size, p=w)
    mu = np.asarray(wl.prompt_mu)[comp]
    sigma = np.asarray(wl.prompt_sigma)[comp]
    return rng.lognormal(mu, sigma) * wl.prompt_scale


def _sample_M(rng: np.random.Generator, wl: Workload, size: int) -> np.ndarray:
    """Vectorized draw of ``size`` M values from the serving model."""
    c = _sample_concurrency(rng, wl, size)
    is_decode = rng.random(size) < wl.p_decode
    prompt = _sample_prompt(rng, wl, size)
    # chunked prefill: a single kernel step never exceeds the token budget.
    prefill = np.minimum(prompt, wl.max_num_batched_tokens - c)
    prefill = np.maximum(prefill, 0.0)
    M = np.where(is_decode, c, np.rint(prefill).astype(np.int64) + c)
    M = np.clip(M, 1, wl.max_num_batched_tokens)
    return M.astype(np.int64)


def _real_nk_weights(fam, gfx: str):
    """Real (N, K) combos and their observed frequencies from the tuned CSV."""
    mnk = _read_real_mnk(fam, gfx)
    vc = mnk.groupby(["N", "K"]).size()
    nk = [(int(n), int(k)) for (n, k) in vc.index]
    w = (vc.to_numpy() / vc.to_numpy().sum()).astype(float)
    return nk, w


def _read_mnk_csv(path: str):
    """Read an arbitrary ``M,N,K`` CSV (tolerating whitespace after commas)."""
    import pandas as pd

    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df[["M", "N", "K"]].astype(int)


def _nk_weights_from_csv(path: str):
    """(N, K) combos + frequencies from an arbitrary ``M,N,K`` CSV.

    Used with ``--nk-csv`` so the weight shapes can be sourced from a file other
    than the tuned CSV -- e.g. an *untuned* CSV whose (N, K) are exactly the
    shapes that miss the tuned table and fall through to the default CK kernel.
    """
    mnk = _read_mnk_csv(path)
    vc = mnk.groupby(["N", "K"]).size()
    nk = [(int(n), int(k)) for (n, k) in vc.index]
    w = (vc.to_numpy() / vc.to_numpy().sum()).astype(float)
    return nk, w


def _mnk_rows_from_csv(path: str) -> set:
    """Set of (M, N, K) tuples in an arbitrary CSV (for exclusion)."""
    mnk = _read_mnk_csv(path)
    return {(int(m), int(n), int(k)) for m, n, k in mnk.itertuples(index=False)}


# --------------------------------------------------------------------------- #
# Stratified-by-regime generation.
#
# Instead of drawing (N,K) from the CSV and M from the raw workload mixture, we
# take an EXPLICIT list of (N,K) weight shapes and, for each, emit a fixed count
# of M values in every serving regime (decode / small-batch / medium- / large-
# prefill). The calibrated workload sampler seeds realistic M where it is rich;
# a deterministic log-spaced grid backs every regime so coverage is guaranteed
# even where the workload almost never lands (e.g. blockscale barely samples the
# decode band). Every emitted (M,N,K) is verified to miss the tuned CSV under
# aiter's runtime M-padding -> it dispatches to the family's DEFAULT CK kernel.
# --------------------------------------------------------------------------- #

def _next_pow2(x: int) -> int:
    """Smallest power of two >= x (matches nextPow2 in csrc/py_itfs_cu/gemm_common.cu)."""
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _padded_m(M: int, N: int, K: int, gl: int) -> int:
    """Replicate aiter's runtime M-padding (getPaddedM, csrc/py_itfs_cu/gemm_common.cu).

    ``get_CKGEMM_config`` probes the merged tuned CSV at gl in {None, 0, 1}:
    gl=None uses M as-is, gl=0 is the fine-grained bucket, gl=1 the coarse one.
    A shape only falls through to the *default* CK kernel when ALL THREE probes
    miss, so novelty must be checked against every padded M -- not just M.
    """
    if gl == 0:
        if M <= 256:
            return (M + 15) // 16 * 16
        if M <= 1024:
            return (M + 31) // 32 * 32
        if M <= 4096:
            return (M + 63) // 64 * 64
        return (M + 127) // 128 * 128
    if gl == 1:
        if M > 8192 and N > 4096:
            return 8192
        return _next_pow2(M)
    return M


def _tuned_M_by_nk(fam, gfx: str, cu_num: int) -> dict:
    """{(N,K): set(M)} tuned for exactly (gfx, cu_num) across the family's CSVs.

    Mirrors what aiter loads at runtime: the base tuned CSV plus matching
    model_configs, keyed on (gfx, cu_num, M, N, K). Rows lacking a matching
    gfx/cu_num never key into that lookup, so they are dropped here too.
    """
    import pandas as pd

    out: dict = {}
    for path in family_csv_files(fam):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"M", "N", "K"}.issubset(df.columns):
            continue
        if "gfx" in df.columns:
            df = df[df["gfx"] == gfx]
        if "cu_num" in df.columns:
            df = df[df["cu_num"] == cu_num]
        if df.empty:
            continue
        for n, k, m in zip(df["N"].astype(int), df["K"].astype(int),
                           df["M"].astype(int)):
            out.setdefault((int(n), int(k)), set()).add(int(m))
    return out


def _hits_default_kernel(m: int, n: int, k: int, tuned_M: set) -> bool:
    """True iff (m,n,k) misses the tuned CSV under EVERY M-padding => default CK kernel."""
    pads = {m, _padded_m(m, n, k, 0), _padded_m(m, n, k, 1)}
    return pads.isdisjoint(tuned_M)


def _log_grid(lo: int, hi: int, count: int) -> list:
    """Deterministic log-spaced integer backbone spanning [lo, hi] (guarantees coverage)."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return [lo]
    xs = np.exp(np.linspace(math.log(lo), math.log(hi), max(count * 4, count)))
    xs = np.unique(np.rint(xs).astype(np.int64))
    xs = xs[(xs >= lo) & (xs <= hi)]
    return xs.tolist()


def _select_spread(vals, count: int) -> list:
    """Pick <=count values spread evenly across sorted-unique ``vals`` (deterministic)."""
    vals = sorted({int(v) for v in vals})
    if len(vals) <= count:
        return vals
    idx = np.unique(np.rint(np.linspace(0, len(vals) - 1, count)).astype(int))
    return [vals[i] for i in idx]


def parse_nk(spec: str) -> list:
    """Parse '51200,5120;5120,25600;...' into [(N,K), ...]."""
    pairs = []
    for chunk in spec.replace(" ", "").split(";"):
        if not chunk:
            continue
        n_str, k_str = chunk.split(",")
        pairs.append((int(n_str), int(k_str)))
    return pairs


def parse_regimes(spec: str) -> list:
    """Parse 'name:lo:hi,name:lo:hi' into [(name, lo, hi), ...]."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, lo, hi = chunk.split(":")
        out.append((name, int(lo), int(hi)))
    return out


def default_regimes(wl: Workload) -> list:
    """decode / small-batch / medium-prefill / large-prefill (top capped at budget B)."""
    cap = int(wl.max_num_batched_tokens)
    return [
        ("decode", 1, 16),
        ("small_batch", 17, 256),
        ("medium_prefill", 257, 4096),
        ("large_prefill", 4097, max(4098, cap)),
    ]


def generate_stratified(fam, wl: Workload, nk_list: list, regimes: list,
                        per_regime: int, seed: int, gfx: str, cu_num: int):
    """For every (N,K), emit ``per_regime`` novel M values in each regime.

    Returns (rows, report) where report maps (N,K,regime_name) -> count emitted.
    """
    rng = np.random.default_rng(seed)
    existing = load_existing(fam, gfx)
    tuned_by_nk = _tuned_M_by_nk(fam, gfx, cu_num)

    # One big workload draw to harvest realistic M per regime (filtered below).
    pool = _sample_M(rng, wl, max(200_000, per_regime * len(regimes) * 4000))

    rows: list = []
    report: dict = {}
    for (n, k) in nk_list:
        tuned_M = tuned_by_nk.get((n, k), set())
        for (rname, lo, hi) in regimes:
            cand: list = []
            seen: set = set()

            def _try_add(m: int) -> None:
                m = int(m)
                if m in seen or not (lo <= m <= hi):
                    return
                if m < fam.min_m or n < fam.min_n or k < fam.min_k:
                    return
                if (m, n, k) in existing:
                    return
                if not _hits_default_kernel(m, n, k, tuned_M):
                    return
                seen.add(m)
                cand.append(m)

            sampled = pool[(pool >= lo) & (pool <= hi)]
            rng.shuffle(sampled)
            for m in sampled.tolist():                     # realistic workload M first
                _try_add(m)
            for m in _log_grid(lo, hi, per_regime * 4):    # deterministic coverage backbone
                _try_add(m)

            chosen = _select_spread(cand, per_regime)
            report[(n, k, rname)] = len(chosen)
            rows.extend((m, n, k) for m in chosen)

    rows = sorted(set(rows), key=lambda r: (r[1], r[2], r[0]))  # by N, K, M
    return rows, report


def verify_stratified(fam, rows, regimes, nk_list, gfx, cu_num, report):
    """Assert disjointness + default-kernel guarantee; print regime coverage."""
    existing = load_existing(fam, gfx)
    overlap = [r for r in rows if r in existing]
    assert not overlap, f"{fam.key}: {len(overlap)} rows overlap the tuned CSV!"
    assert len(rows) == len(set(rows)), "duplicate rows in output!"

    tuned_by_nk = _tuned_M_by_nk(fam, gfx, cu_num)
    bad = [r for r in rows
           if not _hits_default_kernel(r[0], r[1], r[2],
                                       tuned_by_nk.get((r[1], r[2]), set()))]
    assert not bad, (f"{fam.key}: {len(bad)} rows would resolve to a TUNED entry "
                     f"(not the default kernel), e.g. {bad[:5]}")

    print(f"\n  rows                    : {len(rows)}")
    print(f"  default-kernel guarantee: all {len(rows)} rows miss the tuned CSV under "
          f"M-padding on ({gfx}, cu={cu_num})")
    print("  --- per (N,K) x regime coverage ---")
    for (n, k) in nk_list:
        counts = "  ".join(f"{rn}={report.get((n, k, rn), 0)}"
                           for (rn, _, _) in regimes)
        ms = sorted(r[0] for r in rows if r[1] == n and r[2] == k)
        print(f"    (N={n:6d}, K={k:6d}) total={len(ms):3d}  [{counts}]")
        print(f"        M = {ms}")


# --------------------------------------------------------------------------- #
# Novel-(N,K) generation.
#
# Emit a dataset whose (N,K) PAIRS are absent from every dense a8w8_blockscale
# tuned config (base + model_configs; bpreshuffle/fmoe are already filtered out
# by common.families mc_exclude). aiter's dispatch keys on the full (N,K), so a
# pair that appears in NO tuned config misses under every M-padding -> the shape
# runs on the family's DEFAULT / backup CK kernel. M is still modeled from the
# family's calibrated serving workload.
# --------------------------------------------------------------------------- #

def _seen_pairs_and_values(fam, gfx: str):
    """Excluded (N,K) pairs + unique N and K value sets from the family's dense
    tuned configs. ``load_nk_pool`` unions across gfx for a strict exclusion set."""
    pairs = set(load_nk_pool(fam))
    seen_N = sorted({n for (n, _k) in pairs})
    seen_K = sorted({k for (_n, k) in pairs})
    return pairs, seen_N, seen_K


def _novel_nk_candidates(fam, novelty: str, seen_pairs: set,
                         seen_N: list, seen_K: list) -> list:
    """Candidate (N,K) whose PAIR is absent from every tuned config.

    combo  : cross seen N x seen K, drop pairs already tuned (default; individual
             N/K values are reused, only the pairing is new).
    values : N,K restricted to mods OUTSIDE the seen value sets (stronger novelty
             -- neither dimension has ever been tuned).
    Both respect the family's divisibility (n_mod/k_mod) and keep N,K within the
    observed ranges (N >= smallest tuned N) so shapes stay runnable on the CK path.
    """
    min_n = max(fam.min_n, min(seen_N))
    min_k = max(fam.min_k, min(seen_K))
    if novelty == "combo":
        return [(n, k) for n in seen_N for k in seen_K
                if (n, k) not in seen_pairs and n >= min_n and k >= min_k
                and n % fam.n_mod == 0 and k % fam.k_mod == 0]
    # values: unseen individual dims within the observed ranges (bounded cross).
    seenN, seenK = set(seen_N), set(seen_K)
    new_N = _select_spread([n for n in range(min_n, max(seen_N) + 1, fam.n_mod)
                            if n not in seenN], 48)
    new_K = _select_spread([k for k in range(min_k, max(seen_K) + 1, fam.k_mod)
                            if k not in seenK], 48)
    return [(n, k) for n in new_N for k in new_K]


def generate_novel_nk(fam, wl: Workload, nk_list: list, n_points: int, seed: int,
                      gfx: str = TARGET_GFX):
    """Draw ``n_points`` novel (M,N,K) rows over the given novel (N,K) pool.

    M is sampled from the family's calibrated serving workload; (N,K) uniformly
    from ``nk_list``. Rows are unique and disjoint from the tuned CSV. Because
    every (N,K) pair is novel, all rows dispatch to the DEFAULT CK kernel.
    """
    rng = np.random.default_rng(seed)
    existing = load_existing(fam, gfx)
    nk = list(nk_list)
    w = np.full(len(nk), 1.0 / len(nk))
    nk_idx = np.arange(len(nk))
    rows: list = []
    chosen: set = set()
    guard = 0
    guard_max = n_points * 200 + 20000
    batch = max(n_points, 4096)
    while len(rows) < n_points and guard < guard_max:
        guard += 1
        size = min(batch, (n_points - len(rows)) * 4 + 64)
        Ms = _sample_M(rng, wl, size)
        idxs = rng.choice(nk_idx, size=size, p=w)
        for m, gi in zip(Ms.tolist(), idxs.tolist()):
            n, k = nk[gi]
            if m < fam.min_m or n < fam.min_n or k < fam.min_k:
                continue
            key = (m, n, k)
            if key in existing or key in chosen:
                continue
            chosen.add(key)
            rows.append(key)
            if len(rows) >= n_points:
                break
    if len(rows) < n_points:
        raise RuntimeError(
            f"{fam.key}: only produced {len(rows)}/{n_points} novel rows "
            f"(increase --n-nk or widen the M range).")
    rows.sort(key=lambda r: (r[1], r[2], r[0]))
    return rows


def verify_novel_nk(fam, rows, nk_list, seen_pairs, gfx, cu_num):
    """Assert (N,K)-pair novelty, disjointness, and the default-kernel guarantee."""
    import collections

    existing = load_existing(fam, gfx)
    overlap = [r for r in rows if r in existing]
    assert not overlap, f"{fam.key}: {len(overlap)} rows overlap the tuned CSV!"
    assert len(rows) == len(set(rows)), "duplicate rows in output!"

    used_nk = {(n, k) for (_m, n, k) in rows}
    bad_pairs = used_nk & seen_pairs
    assert not bad_pairs, (f"{fam.key}: chosen (N,K) appear in a tuned config: "
                           f"{sorted(bad_pairs)[:5]}")

    tuned_by_nk = _tuned_M_by_nk(fam, gfx, cu_num)
    bad = [r for r in rows
           if not _hits_default_kernel(r[0], r[1], r[2],
                                       tuned_by_nk.get((r[1], r[2]), set()))]
    assert not bad, (f"{fam.key}: {len(bad)} rows would resolve to a TUNED entry "
                     f"(not the default kernel), e.g. {bad[:5]}")

    sM = np.array([r[0] for r in rows])
    print("\n  overlap with tuned CSV  : 0  (verified disjoint)")
    print(f"  (N,K) pair novelty      : all {len(used_nk)} shapes absent from every "
          f"a8w8_blockscale tuned config")
    print(f"  default-kernel guarantee: all {len(rows)} rows miss the tuned CSV under "
          f"M-padding on ({gfx}, cu={cu_num})")
    print(f"  rows                    : {len(rows)}")
    print(f"  M distribution          : med={int(np.median(sM))}  "
          f"p90={int(np.quantile(sM, .9))}  max={int(sM.max())}  "
          f"frac<=512={(sM <= 512).mean():.3f}")
    sv = collections.Counter((r[1], r[2]) for r in rows)
    print("  --- per (N,K) row counts ---")
    for (n, k) in nk_list:
        print(f"    (N={n:6d}, K={k:6d})  rows={sv.get((n, k), 0)}")


def generate(fam, wl: Workload, n_points: int, seed: int, gfx: str = TARGET_GFX,
             nk_csv: str | None = None):
    rng = np.random.default_rng(seed)
    existing = load_existing(fam, gfx)          # (M,N,K) tuples to avoid
    if nk_csv:
        # Source weight shapes from the given CSV (e.g. untuned), and also treat
        # its own (M,N,K) rows as existing so we emit *novel* M regimes for them.
        nk, w = _nk_weights_from_csv(nk_csv)
        existing = existing | _mnk_rows_from_csv(nk_csv)
    else:
        nk, w = _real_nk_weights(fam, gfx)
    nk_idx = np.arange(len(nk))

    rows: list[tuple] = []
    chosen: set[tuple] = set()
    guard = 0
    guard_max = n_points * 200 + 20000
    batch = max(n_points, 4096)
    while len(rows) < n_points and guard < guard_max:
        guard += 1
        need = n_points - len(rows)
        size = min(batch, need * 4 + 64)
        Ms = _sample_M(rng, wl, size)
        idxs = rng.choice(nk_idx, size=size, p=w)
        for m, gi in zip(Ms.tolist(), idxs.tolist()):
            n, k = nk[gi]
            if m < fam.min_m or n < fam.min_n or k < fam.min_k:
                continue
            key = (m, n, k)
            if key in existing or key in chosen:
                continue
            chosen.add(key)
            rows.append(key)
            if len(rows) >= n_points:
                break

    if len(rows) < n_points:
        raise RuntimeError(
            f"{fam.key}: only produced {len(rows)}/{n_points} novel points "
            f"(space exhausted for the sampled (N,K) x M range)."
        )
    rows.sort(key=lambda r: (r[1], r[2], r[0]))   # by N, K, M
    return rows


def verify(fam, rows, wl: Workload, gfx: str = TARGET_GFX,
           nk_csv: str | None = None):
    """Print a real-vs-synthetic comparison and assert disjointness."""
    existing = load_existing(fam, gfx)
    if nk_csv:
        existing = existing | _mnk_rows_from_csv(nk_csv)
    overlap = sum(1 for r in rows if r in existing)
    assert overlap == 0, f"{fam.key}: {overlap} synthetic rows overlap the tuned/nk CSV!"
    assert len(rows) == len(set(rows)), "duplicate rows in output!"

    real = _read_real_mnk(fam, gfx)
    rM = real["M"].to_numpy()
    sM = np.array([r[0] for r in rows])
    print("\n  overlap with tuned CSV : 0  (verified disjoint)")
    print(f"  rows                   : real={len(rM)}  synth={len(sM)}")

    def line(tag, a):
        f_dec = (a <= _DECODE_MAX).mean()
        print(f"  {tag:6s} med={int(np.median(a)):6d}  p90={int(np.quantile(a,.9)):6d}  "
              f"p99={int(np.quantile(a,.99)):6d}  max={int(a.max()):6d}  "
              f"frac<=512={f_dec:.3f}")
    print("  --- M distribution ---")
    line("real", rM)
    line("synth", sM)

    # per-(N,K) frequency reproduction (vs the sampled source's target weights)
    print("  --- (N,K) frequency  (target -> synth) ---")
    if nk_csv:
        ref_nk, ref_w = _nk_weights_from_csv(nk_csv)
        ref = dict(zip(ref_nk, ref_w))
    else:
        rv = real.groupby(["N", "K"]).size()
        ref = {(int(n), int(k)): c / len(rM) for (n, k), c in rv.items()}
    import collections
    sv = collections.Counter((r[1], r[2]) for r in rows)
    for (n, k), rf in sorted(ref.items()):
        sc = sv.get((int(n), int(k)), 0)
        print(f"    ({int(n)},{int(k)}): {rf:.3f} -> {sc/len(sM):.3f}")


def write_csv(fam, rows, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("M", "N", "K"))
        w.writerows(rows)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="a8w8_blockscale")
    ap.add_argument("--gfx", default=TARGET_GFX)
    ap.add_argument("--n", type=int, default=None,
                    help="rows to generate (default: match the real gfx row count)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output CSV (default: datasets/<family>_gfx950_synth.csv)")
    # workload knob overrides (any left unset stay calibrated to the CSV). The
    # prompt-length *shape* is always calibrated; --prompt-scale shifts the whole
    # distribution for "what if prompts were 2x longer?" style exploration.
    ap.add_argument("--p-decode", type=float, default=None)
    ap.add_argument("--prompt-scale", type=float, default=None,
                    help="multiply all prompt lengths (1.0 = calibrated)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="concurrency cap S (also scales mean concurrency)")
    ap.add_argument("--conc-a", type=float, default=2.0)
    ap.add_argument("--conc-b", type=float, default=3.0)
    ap.add_argument("--n-modes", type=int, default=4,
                    help="number of prompt-length (context) mixture modes")
    ap.add_argument("--nk-csv", default=None,
                    help="source (N,K) weight shapes from this M,N,K CSV instead "
                         "of the tuned CSV (its own rows are excluded so only "
                         "novel M regimes are emitted). M is still modeled from "
                         "the family's calibrated serving workload.")
    # stratified-by-M-regime mode -------------------------------------------- #
    ap.add_argument("--stratified", action="store_true",
                    help="emit --per-regime M values in EACH serving regime for "
                         "every --nk (N,K); all rows are verified to hit the "
                         "family's DEFAULT CK kernel on (gfx, cu-num).")
    ap.add_argument("--nk", default=None,
                    help="explicit weight shapes 'N,K;N,K;...' for --stratified.")
    ap.add_argument("--per-regime", type=int, default=12,
                    help="M values per regime per (N,K) in --stratified mode.")
    ap.add_argument("--regimes", default=None,
                    help="override regimes as 'name:lo:hi,...' (default: decode / "
                         "small_batch / medium_prefill / large_prefill).")
    ap.add_argument("--cu-num", type=int, default=256,
                    help="target compute-unit count for the default-kernel "
                         "guarantee (gfx950 MI355X = 256).")
    # novel-(N,K) mode ------------------------------------------------------- #
    ap.add_argument("--novel-nk", action="store_true",
                    help="emit a dataset whose (N,K) PAIRS are absent from every "
                         "dense a8w8_blockscale tuned config (configs + "
                         "model_configs) -> every row dispatches to the DEFAULT "
                         "CK kernel. M is still modeled from the serving workload.")
    ap.add_argument("--n-nk", type=int, default=12,
                    help="number of distinct novel (N,K) shapes for --novel-nk.")
    ap.add_argument("--nk-novelty", choices=["combo", "values"], default="combo",
                    help="combo (default): novel (N,K) pairs from seen N x seen K "
                         "minus seen pairs; values: N,K from mods OUTSIDE the seen "
                         "N/K value sets (stronger novelty).")
    args = ap.parse_args()

    fam = get_family(args.family)
    wl = fit_workload(fam, args.gfx, conc_a=args.conc_a, conc_b=args.conc_b,
                      n_modes=args.n_modes)
    # apply overrides
    for knob in ("p_decode", "prompt_scale", "max_num_batched_tokens", "max_num_seqs"):
        v = getattr(args, knob)
        if v is not None:
            setattr(wl, knob, v)

    real_n = len(_read_real_mnk(fam, args.gfx))
    n = args.n if args.n is not None else real_n
    default_tag = "untuned_nk_synth" if args.nk_csv else "synth"
    out = args.out or os.path.join(
        _ROOT, "datasets", f"{fam.key}_gfx950_{default_tag}.csv")

    print(f"Workload-driven synthetic dataset for '{fam.key}' (gfx={args.gfx})")
    print(f"  M calibrated to: {fam.tuned_csv}  ({real_n} real rows)")
    if args.nk_csv:
        nk_src, nk_w = _nk_weights_from_csv(args.nk_csv)
        print(f"  (N,K) sourced from: {os.path.relpath(args.nk_csv)}  "
              f"({len(nk_src)} distinct shapes)")
    print()
    print("  fitted / active knobs:")
    for kk in ("p_decode", "prompt_scale", "max_num_batched_tokens",
               "max_num_seqs", "conc_a", "conc_b"):
        vv = getattr(wl, kk)
        vv = round(vv, 4) if isinstance(vv, float) else vv
        print(f"    {kk:24s} = {vv}")
    print(f"    {'-> mean concurrency E[c]':24s} = {wl.concurrency_mean():.1f}")
    print(f"    {'-> prompt-length modes':24s} (context-length mixture):")
    for wgt, med, sg in sorted(zip(wl.prompt_w, wl.mode_medians(), wl.prompt_sigma),
                               key=lambda t: t[1]):
        print(f"        {wgt:5.0%} of requests  ~ median {int(med):6d} tok  (sigma={sg:.2f})")

    if args.novel_nk:
        seen_pairs, seen_N, seen_K = _seen_pairs_and_values(fam, args.gfx)
        cands = sorted(set(_novel_nk_candidates(fam, args.nk_novelty, seen_pairs,
                                                seen_N, seen_K)),
                       key=lambda nk: (nk[0] * nk[1], nk[0], nk[1]))
        if not cands:
            ap.error("no novel (N,K) candidates available for the chosen novelty mode")
        if args.n_nk < len(cands):
            idx = np.unique(np.rint(np.linspace(0, len(cands) - 1, args.n_nk)).astype(int))
            nk_list = [cands[i] for i in idx]
        else:
            nk_list = cands
        nnk_out = args.out or os.path.join(
            _ROOT, "datasets", f"{fam.key}_gfx950_novel_nk.csv")
        print(f"\n  novel-nk mode: novelty={args.nk_novelty}  "
              f"seen pairs={len(seen_pairs)} (N:{len(seen_N)} x K:{len(seen_K)})  "
              f"candidates={len(cands)}  chosen {len(nk_list)} (N,K):")
        for (pn, pk) in nk_list:
            print(f"      (N={pn:6d}, K={pk:6d})   N*K={pn * pk}")
        rows = generate_novel_nk(fam, wl, nk_list, n, args.seed, args.gfx)
        verify_novel_nk(fam, rows, nk_list, seen_pairs, args.gfx, args.cu_num)
        path = write_csv(fam, rows, nnk_out)
        print(f"\n  wrote {len(rows)} rows -> {os.path.relpath(path, _ROOT)}")
        return 0

    if args.stratified:
        if not args.nk:
            ap.error("--stratified requires --nk 'N,K;N,K;...'")
        nk_list = parse_nk(args.nk)
        regimes = parse_regimes(args.regimes) if args.regimes else default_regimes(wl)
        strat_out = args.out or os.path.join(
            _ROOT, "datasets", f"{fam.key}_gfx950_stratified.csv")
        print(f"\n  stratified mode: {args.per_regime} M / regime / (N,K)  "
              f"on (gfx={args.gfx}, cu={args.cu_num})")
        print("    regimes (M bands): "
              + ", ".join(f"{rn}[{lo}..{hi}]" for (rn, lo, hi) in regimes))
        print(f"    (N,K) shapes    : {nk_list}")
        rows, report = generate_stratified(
            fam, wl, nk_list, regimes, args.per_regime, args.seed,
            args.gfx, args.cu_num)
        verify_stratified(fam, rows, regimes, nk_list, args.gfx, args.cu_num, report)
        path = write_csv(fam, rows, strat_out)
        print(f"\n  wrote {len(rows)} rows -> {os.path.relpath(path, _ROOT)}")
        return 0

    rows = generate(fam, wl, n, args.seed, args.gfx, nk_csv=args.nk_csv)
    verify(fam, rows, wl, args.gfx, nk_csv=args.nk_csv)
    path = write_csv(fam, rows, out)
    print(f"\n  wrote {len(rows)} rows -> {os.path.relpath(path, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
