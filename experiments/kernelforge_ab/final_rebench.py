#!/usr/bin/env python3
"""Same-session alternating re-benchmark of each final A/B kernel."""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
USER = "15724:15724"
ROUNDS = 3


def cid(item: dict) -> str:
    return f"{item['shape_id']}-{item['arm_id']}"


def run_one(item: dict) -> dict:
    campaign = cid(item)
    meta = json.loads((HERE / "runs" / campaign / "campaign.json").read_text())
    worktree = Path(meta["worktree"])
    cmd = [
        "docker",
        "exec",
        "-u",
        USER,
        "-e",
        "HOME=/home/demantri",
        "-e",
        "USER=demantri",
        "-e",
        "LOGNAME=demantri",
        "-e",
        f"KF_AITER_ROOT={worktree}",
        "origami-dev",
        "python3",
        str(worktree / "forge_driver.py"),
        "--shape",
        f"M={item['M']},N={item['N']},K={item['K']}",
        "--warmup",
        "30",
        "--iters",
        "100",
        "--bench-mode",
    ]
    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
    text = proc.stdout + "\n" + proc.stderr
    match = re.search(r"median_ms:\s*([\d.]+)", text)
    return {
        "campaign_id": campaign,
        "shape_id": item["shape_id"],
        "start_kind": item["start_kind"],
        "returncode": proc.returncode,
        "median_ms": float(match.group(1)) if match else None,
        "elapsed_sec": round(time.time() - started, 3),
        "tail": text[-1200:],
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    by_shape: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in manifest["campaigns"]:
        by_shape[item["shape_id"]][item["start_kind"]] = item

    observations = []
    for round_id in range(ROUNDS):
        order = ["origami", "default"] if round_id % 2 == 0 else ["default", "origami"]
        for shape_id in sorted(by_shape):
            for kind in order:
                item = by_shape[shape_id][kind]
                print(f"round={round_id + 1} {cid(item)}", flush=True)
                result = run_one(item)
                result["round"] = round_id + 1
                observations.append(result)
                print(
                    f"  rc={result['returncode']} median_ms={result['median_ms']} "
                    f"elapsed={result['elapsed_sec']}s",
                    flush=True,
                )
    (HERE / "final_rebench_observations.json").write_text(
        json.dumps(observations, indent=2) + "\n"
    )

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in observations:
        if row["returncode"] == 0 and row["median_ms"] is not None:
            grouped[row["campaign_id"]].append(row["median_ms"])
    summary = []
    for item in manifest["campaigns"]:
        values = grouped.get(cid(item), [])
        summary.append(
            {
                "campaign_id": cid(item),
                "shape_id": item["shape_id"],
                "start_kind": item["start_kind"],
                "M": item["M"],
                "N": item["N"],
                "K": item["K"],
                "rounds": len(values),
                "median_ms": statistics.median(values) if values else None,
                "min_ms": min(values) if values else None,
                "max_ms": max(values) if values else None,
            }
        )
    fields = list(summary[0])
    with (HERE / "final_rebench.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(summary)
    (HERE / "final_rebench.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(row["rounds"] == ROUNDS for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
