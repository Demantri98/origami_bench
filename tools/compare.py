#!/usr/bin/env python3
"""Summarize the prod comparison: aiter default fallback vs Origami-picked kernel.

Reads ``results/<family>_aiter_prod.csv`` (from aiter_run.py in prod mode with
the Origami backup enabled) and reports, per family:

  * how often the Origami-selected CK kernel beats aiter's built-in default
    fallback (win rate),
  * mean speedup of Origami's pick over the default,
  * mean measured TFLOPS for each,
  * (if a sweep oracle file exists) how much of the exhaustive-search headroom
    Origami's cheap pick recovers.

Also emits a tidy joined ``results/<family>_compare.csv``. Pure-python; runs on
the host.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from common.families import FAMILIES, Family, get_family  # noqa: E402


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _avg(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def compare_family(fam: Family, results_dir: str) -> str | None:
    prod_path = os.path.join(results_dir, f"{fam.key}_aiter_prod.csv")
    if not os.path.exists(prod_path):
        print(f"  [{fam.key}] no {os.path.basename(prod_path)} (run aiter_run.py --mode prod)")
        return None
    rows = list(csv.DictReader(open(prod_path, newline="")))

    # Optional predicted-cycles (origami) and oracle (sweep) side files.
    def _index(path, keycols=("M", "N", "K", "q_dtype_w")):
        if not os.path.exists(path):
            return {}
        return {tuple(r.get(c, "") for c in keycols): r
                for r in csv.DictReader(open(path, newline=""))}
    origami_pred = _index(os.path.join(results_dir, f"{fam.key}_origami.csv"))
    sweep = _index(os.path.join(results_dir, f"{fam.key}_aiter_both.csv")) or \
        _index(os.path.join(results_dir, f"{fam.key}_aiter_sweep.csv"))

    out_path = os.path.join(results_dir, f"{fam.key}_compare.csv")
    cols = ["family", "M", "N", "K", "q_dtype_w",
            "prod_default_us", "prod_default_tflops",
            "origami_kernelId", "origami_us", "origami_tflops",
            "origami_correct", "origami_err_ratio",
            "origami_pred_cycles", "winner", "origami_speedup_vs_prod",
            "sweep_best_us", "origami_frac_of_oracle"]

    speedups, wins, both_ok, mismatches = [], 0, 0, 0
    prod_tf, orig_tf = [], []
    frac_oracle = []
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            key = (r.get("M"), r.get("N"), r.get("K"), r.get("q_dtype_w", ""))
            p_us, o_us = _num(r.get("prod_us")), _num(r.get("origami_us"))
            sp = _num(r.get("origami_speedup_vs_prod"))
            s_us = _num(sweep.get(key, {}).get("sweep_best_us"))
            # Correctness gate: only trust the Origami pick when it matched prod.
            correct_field = r.get("origami_correct", "")
            correct = correct_field not in ("False", "0")
            if o_us and correct_field == "False":
                mismatches += 1
            frac = ""
            # fraction of oracle headroom recovered (1.0 == matches best kernel)
            if correct and p_us and o_us and s_us and p_us > s_us:
                frac = round((p_us - o_us) / (p_us - s_us), 3)
                frac_oracle.append(frac)
            if p_us and o_us and correct:
                both_ok += 1
                speedups.append(sp if sp is not None else round(p_us / o_us, 3))
                if o_us < p_us:
                    wins += 1
            if _num(r.get("prod_tflops")) is not None:
                prod_tf.append(_num(r.get("prod_tflops")))
            if _num(r.get("origami_tflops")) is not None:
                orig_tf.append(_num(r.get("origami_tflops")))
            w.writerow({
                "family": fam.key, "M": key[0], "N": key[1], "K": key[2], "q_dtype_w": key[3],
                "prod_default_us": r.get("prod_us", ""), "prod_default_tflops": r.get("prod_tflops", ""),
                "origami_kernelId": r.get("origami_kernelId", ""),
                "origami_us": r.get("origami_us", ""), "origami_tflops": r.get("origami_tflops", ""),
                "origami_correct": r.get("origami_correct", ""),
                "origami_err_ratio": r.get("origami_err_ratio", ""),
                "origami_pred_cycles": origami_pred.get(key, {}).get("pred_latency_cycles", ""),
                "winner": r.get("winner", ""),
                "origami_speedup_vs_prod": r.get("origami_speedup_vs_prod", ""),
                "sweep_best_us": sweep.get(key, {}).get("sweep_best_us", ""),
                "origami_frac_of_oracle": frac,
            })

    def _geomean(xs):
        xs = [x for x in xs if x and x > 0]
        return round(math.exp(sum(math.log(x) for x in xs) / len(xs)), 3) if xs else None

    def _median(xs):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[len(xs) // 2], 3) if xs else None

    print(f"  [{fam.key}] {len(rows)} shapes, {both_ok} correct & measured "
          f"({mismatches} failed correctness) -> {out_path}")
    if both_ok:
        # geomean/median of speedup ratios (arithmetic mean of ratios is biased).
        print(f"      Origami beats default: {wins}/{both_ok} ({100*wins/both_ok:.0f}%)  "
              f"speedup geomean={_geomean(speedups)} median={_median(speedups)}")
    print(f"      mean TFLOPS  default={_avg(prod_tf)}  origami={_avg(orig_tf)}"
          + (f"  |  geomean frac-of-oracle={_geomean(frac_oracle)}" if frac_oracle else ""))
    return {
        "family": fam.key, "shapes": len(rows), "correct_measured": both_ok,
        "correctness_failures": mismatches,
        "origami_win_pct": round(100 * wins / both_ok, 1) if both_ok else "",
        "speedup_geomean": _geomean(speedups), "speedup_median": _median(speedups),
        "mean_tflops_default": _avg(prod_tf), "mean_tflops_origami": _avg(orig_tf),
        "frac_of_oracle_geomean": _geomean(frac_oracle) if frac_oracle else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", nargs="*", default=list(FAMILIES))
    ap.add_argument("--results-dir", default=os.path.join(_ROOT, "results"))
    args = ap.parse_args()
    print("aiter default fallback vs Origami-picked kernel\n")
    stats = []
    for key in args.family:
        s = compare_family(get_family(key), args.results_dir)
        if s:
            stats.append(s)
    if stats:
        summary_path = os.path.join(args.results_dir, "summary.csv")
        with open(summary_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
            w.writeheader()
            w.writerows(stats)
        print(f"\nAggregate summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
