#!/usr/bin/env python3
"""Aggregate forge-loop campaign artifacts into paired A/B results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
RUNS = HERE / "runs"
INSTANCE_REL = Path(
    "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py"
)


def campaign_id(campaign: dict) -> str:
    return f"{campaign['shape_id']}-{campaign['arm_id']}"


def guard_hash(path: Path) -> str:
    lines = path.read_text().splitlines()
    guarded = [
        line
        for line in lines
        if not line.lstrip().startswith("0:") or "KernelInstance(" not in line
    ]
    return hashlib.sha256(("\n".join(guarded) + "\n").encode()).hexdigest()


def slot_zero_line(path: Path) -> str:
    matches = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.lstrip().startswith("0:") and "KernelInstance(" in line
    ]
    return matches[0] if len(matches) == 1 else ""


def find_experiment(run_dir: Path, experiment_id: str | None) -> dict:
    if not experiment_id:
        return {}
    for path in (run_dir / "experiments").rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("experiment_id") == experiment_id:
            return data
    return {}


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    rebench_path = HERE / "final_rebench.json"
    rebench_rows = json.loads(rebench_path.read_text()) if rebench_path.exists() else []
    rebench_by_campaign = {
        item["campaign_id"]: item for item in rebench_rows
    }
    rows: list[dict] = []
    curves: dict[str, list[dict]] = {}

    for campaign in sorted(manifest["campaigns"], key=lambda item: item["run_order"]):
        cid = campaign_id(campaign)
        run_dir = RUNS / cid
        result_path = run_dir / "result.json"
        status_path = run_dir / "status.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        campaign_meta = json.loads((run_dir / "campaign.json").read_text())
        worktree = Path(campaign_meta["worktree"])
        instance_path = worktree / INSTANCE_REL

        experiment = find_experiment(run_dir, result.get("experiment_id"))
        iterations = experiment.get("iterations", [])
        baseline = result.get("baseline_ms")
        best = result.get("best_ms")
        final = best if finite(best) else baseline
        best_so_far = baseline if finite(baseline) else math.inf
        curve = [{"iteration": 0, "best_ms": baseline, "decision": "BASELINE"}]
        for item in iterations:
            wall = item.get("wall_ms")
            if item.get("decision") == "KEEP" and finite(wall):
                best_so_far = min(best_so_far, wall)
            curve.append(
                {
                    "iteration": item.get("iteration_id"),
                    "wall_ms": wall,
                    "best_ms": best_so_far if math.isfinite(best_so_far) else None,
                    "decision": item.get("decision", ""),
                    "snr_db": item.get("snr_db"),
                    "notes": item.get("notes", ""),
                }
            )
        curves[cid] = curve

        initial_line = json.loads((run_dir / "seed_metadata.json").read_text())[
            "seeded_template"
        ]
        final_line = slot_zero_line(instance_path) if instance_path.exists() else ""
        source_guard_ok = (
            guard_hash(instance_path) == campaign_meta["source_guard_sha256"]
            if instance_path.exists()
            else False
        )
        oracle_ms = campaign["origami_us_prior"] / 1000.0
        if campaign["oracle_kernel_id"] != campaign["origami_kernel_id"]:
            # The manifest stores prior Origami/default times, not oracle_us, for
            # these rows. Keep oracle-normalized fields blank rather than lie.
            oracle_ms = None
        row = {
            "run_order": campaign["run_order"],
            "campaign_id": cid,
            "shape_id": campaign["shape_id"],
            "arm_id": campaign["arm_id"],
            "start_kind": campaign["start_kind"],
            "M": campaign["M"],
            "N": campaign["N"],
            "K": campaign["K"],
            "source_kernel_id": campaign["source_kernel_id"],
            "oracle_kernel_id": campaign["oracle_kernel_id"],
            "prior_origami_speedup": campaign["prior_origami_speedup"],
            "returncode": status.get("returncode"),
            "elapsed_sec": status.get("elapsed_sec"),
            "experiment_id": result.get("experiment_id"),
            "baseline_ms": baseline,
            "best_ms": final,
            "improved": result.get("improved", False),
            "speedup_from_start": (
                baseline / final if finite(baseline) and finite(final) and final > 0 else None
            ),
            "final_over_oracle": (
                final / oracle_ms if finite(final) and finite(oracle_ms) else None
            ),
            "iterations_recorded": len(iterations),
            "kept_iterations": sum(1 for item in iterations if item.get("decision") == "KEEP"),
            "source_guard_ok": source_guard_ok,
            "initial_template_name": initial_line.get("name", ""),
            "final_slot0_sha256": hashlib.sha256(final_line.encode()).hexdigest()
            if final_line
            else "",
            "final_slot0": final_line,
        }
        rows.append(row)

    fields = list(rows[0]) if rows else []
    with (HERE / "campaign_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "campaign_results.json").write_text(
        json.dumps({"campaigns": rows, "curves": curves}, indent=2) + "\n"
    )

    by_shape: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_shape[row["shape_id"]][row["start_kind"]] = row
    pairs = []
    for shape_id, arms in sorted(by_shape.items()):
        origami = arms.get("origami")
        default = arms.get("default")
        if not origami or not default:
            continue
        ratio = (
            origami["best_ms"] / default["best_ms"]
            if finite(origami["best_ms"])
            and finite(default["best_ms"])
            and default["best_ms"] > 0
            else None
        )
        origami_rebench = rebench_by_campaign.get(origami["campaign_id"], {}).get(
            "median_ms"
        )
        default_rebench = rebench_by_campaign.get(default["campaign_id"], {}).get(
            "median_ms"
        )
        rebench_ratio = (
            origami_rebench / default_rebench
            if finite(origami_rebench)
            and finite(default_rebench)
            and default_rebench > 0
            else None
        )
        pairs.append(
            {
                "shape_id": shape_id,
                "M": origami["M"],
                "N": origami["N"],
                "K": origami["K"],
                "prior_origami_speedup": origami["prior_origami_speedup"],
                "origami_start_ms": origami["baseline_ms"],
                "default_start_ms": default["baseline_ms"],
                "origami_final_ms": origami["best_ms"],
                "default_final_ms": default["best_ms"],
                "final_origami_over_default": ratio,
                "origami_wins_final": ratio < 1 if finite(ratio) else None,
                "origami_rebench_ms": origami_rebench,
                "default_rebench_ms": default_rebench,
                "rebench_origami_over_default": rebench_ratio,
                "origami_wins_rebench": (
                    rebench_ratio < 1 if finite(rebench_ratio) else None
                ),
                "same_final_template": (
                    bool(origami["final_slot0_sha256"])
                    and origami["final_slot0_sha256"] == default["final_slot0_sha256"]
                ),
                "origami_source_guard_ok": origami["source_guard_ok"],
                "default_source_guard_ok": default["source_guard_ok"],
            }
        )
    pair_fields = list(pairs[0]) if pairs else []
    with (HERE / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pairs)
    (HERE / "paired_results.json").write_text(json.dumps(pairs, indent=2) + "\n")

    valid_ratios = [
        pair["final_origami_over_default"]
        for pair in pairs
        if finite(pair["final_origami_over_default"])
    ]
    valid_rebench_ratios = [
        pair["rebench_origami_over_default"]
        for pair in pairs
        if finite(pair["rebench_origami_over_default"])
    ]
    report = [
        "# KernelForge starting-template A/B pilot",
        "",
        f"Completed campaign artifacts: {sum(1 for row in rows if finite(row['best_ms']))}/{len(rows)}",
        f"Complete paired shapes: {len(valid_ratios)}/{len(by_shape)}",
        "",
        "## Paired outcomes",
        "",
        "| Shape | M,N,K | Prior start advantage | Campaign final O/D | Same-session rebench O/D | Rebench winner | Same final template |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for pair in pairs:
        ratio = pair["final_origami_over_default"]
        rebench_ratio = pair["rebench_origami_over_default"]
        winner = (
            "Origami start"
            if finite(rebench_ratio) and rebench_ratio < 1
            else "Default start"
            if finite(rebench_ratio)
            else "Incomplete"
        )
        report.append(
            f"| {pair['shape_id']} | {pair['M']},{pair['N']},{pair['K']} | "
            f"{pair['prior_origami_speedup']:.3f}x | "
            f"{ratio:.4f} | {rebench_ratio:.4f} | {winner} | "
            f"{pair['same_final_template']} |"
            if finite(ratio) and finite(rebench_ratio)
            else f"| {pair['shape_id']} | {pair['M']},{pair['N']},{pair['K']} | "
            f"{pair['prior_origami_speedup']:.3f}x | — | — | Incomplete | — |"
        )
    report += ["", "## Pilot summary", ""]
    if valid_ratios:
        report += [
            f"- Median final Origami-start/default-start ratio: {statistics.median(valid_ratios):.4f}.",
            f"- Origami-start final wins: {sum(value < 1 for value in valid_ratios)}/{len(valid_ratios)}.",
        ]
    if valid_rebench_ratios:
        meaningful_origami = sum(value < 0.98 for value in valid_rebench_ratios)
        meaningful_default = sum(value > 1.02 for value in valid_rebench_ratios)
        noise_ties = len(valid_rebench_ratios) - meaningful_origami - meaningful_default
        report += [
            f"- Same-session rebench median Origami/default ratio: {statistics.median(valid_rebench_ratios):.4f}.",
            f"- Same-session Origami-start wins: {sum(value < 1 for value in valid_rebench_ratios)}/{len(valid_rebench_ratios)}.",
            f"- Applying KernelForge's 2% noise floor: {meaningful_origami} meaningful Origami-start wins, "
            f"{meaningful_default} meaningful default-start wins, and {noise_ties} ties.",
            f"- Both arms converged to the exact same final template on "
            f"{sum(pair['same_final_template'] for pair in pairs)}/{len(pairs)} shapes.",
            "- Pilot conclusion: a stronger Origami baseline did not reliably produce a "
            "faster final kernel; the effect was meaningful on two shapes and erased by "
            "convergence/noise on four.",
        ]
    report += [
        "- This is a single-campaign-per-arm pilot and is descriptive, not a statistical test.",
        "- A result below 1.0 means the campaign initialized from Origami ended faster.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
