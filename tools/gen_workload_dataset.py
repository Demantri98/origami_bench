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
from common.shapes import family_csv_files, load_existing  # noqa: E402

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


def generate(fam, wl: Workload, n_points: int, seed: int, gfx: str = TARGET_GFX):
    rng = np.random.default_rng(seed)
    existing = load_existing(fam, gfx)          # (M,N,K) tuples to avoid
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


def verify(fam, rows, wl: Workload, gfx: str = TARGET_GFX):
    """Print a real-vs-synthetic comparison and assert disjointness."""
    existing = load_existing(fam, gfx)
    overlap = sum(1 for r in rows if r in existing)
    assert overlap == 0, f"{fam.key}: {overlap} synthetic rows overlap the tuned CSV!"
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

    # per-(N,K) frequency reproduction
    print("  --- (N,K) frequency  (real -> synth) ---")
    rv = real.groupby(["N", "K"]).size()
    import collections
    sv = collections.Counter((r[1], r[2]) for r in rows)
    for (n, k), rc in rv.items():
        sc = sv.get((int(n), int(k)), 0)
        print(f"    ({int(n)},{int(k)}): {rc/len(rM):.3f} -> {sc/len(sM):.3f}")


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
    out = args.out or os.path.join(_ROOT, "datasets", f"{fam.key}_gfx950_synth.csv")

    print(f"Workload-driven synthetic dataset for '{fam.key}' (gfx={args.gfx})")
    print(f"  calibrated to: {fam.tuned_csv}  ({real_n} real rows)\n")
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

    rows = generate(fam, wl, n, args.seed, args.gfx)
    verify(fam, rows, wl, args.gfx)
    path = write_csv(fam, rows, out)
    print(f"\n  wrote {len(rows)} rows -> {os.path.relpath(path, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
