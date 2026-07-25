#!/usr/bin/env python3
"""Build, correctness-check, and benchmark every seeded pilot baseline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
USER = "15724:15724"


def campaign_id(campaign: dict) -> str:
    return f"{campaign['shape_id']}-{campaign['arm_id']}"


def driver_cmd(campaign: dict, extra: list[str]) -> list[str]:
    cid = campaign_id(campaign)
    meta = json.loads((HERE / "runs" / cid / "campaign.json").read_text())
    worktree = Path(meta["worktree"])
    return [
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
        f"M={campaign['M']},N={campaign['N']},K={campaign['K']}",
        *extra,
    ]


def run_driver(campaign: dict, extra: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        driver_cmd(campaign, extra),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = {value.strip() for value in args.only.split(",") if value.strip()}
    manifest = json.loads(MANIFEST.read_text())
    failures = 0

    for campaign in sorted(manifest["campaigns"], key=lambda item: item["run_order"]):
        cid = campaign_id(campaign)
        if selected and cid not in selected:
            continue
        out_path = HERE / "runs" / cid / "preflight.json"
        if out_path.exists() and not args.force:
            print(f"{cid}: preflight exists, skipping", flush=True)
            continue
        started = time.time()
        print(
            f"{cid}: build/correctness M={campaign['M']} N={campaign['N']} K={campaign['K']}",
            flush=True,
        )
        check = run_driver(campaign, ["--mode", "smoke"], timeout=600)
        check_text = check.stdout + "\n" + check.stderr
        snr_match = re.search(r"SNR:\s*([\d.]+)\s*dB", check_text)
        snr = float(snr_match.group(1)) if snr_match else None
        print(f"{cid}: correctness rc={check.returncode} SNR={snr}", flush=True)

        bench = None
        median_ms = None
        if check.returncode == 0 and snr is not None and snr >= 30:
            bench = run_driver(
                campaign,
                ["--warmup", "100", "--iters", "100", "--bench-mode"],
                timeout=180,
            )
            bench_text = bench.stdout + "\n" + bench.stderr
            median_match = re.search(r"median_ms:\s*([\d.]+)", bench_text)
            median_ms = float(median_match.group(1)) if median_match else None
            print(
                f"{cid}: benchmark rc={bench.returncode} median_ms={median_ms}",
                flush=True,
            )
        prior_us = (
            campaign["origami_us_prior"]
            if campaign["start_kind"] == "origami"
            else campaign["default_us_prior"]
        )
        prior_ms = prior_us / 1000.0
        ratio_to_prior = median_ms / prior_ms if median_ms is not None else None
        ok = (
            check.returncode == 0
            and snr is not None
            and snr >= 30
            and bench is not None
            and bench.returncode == 0
            and median_ms is not None
            and 0.5 <= ratio_to_prior <= 1.5
        )
        result = {
            "campaign_id": cid,
            "ok": ok,
            "correctness_returncode": check.returncode,
            "snr_db": snr,
            "benchmark_returncode": bench.returncode if bench else None,
            "median_ms": median_ms,
            "prior_ms": prior_ms,
            "ratio_to_prior": ratio_to_prior,
            "elapsed_sec": round(time.time() - started, 3),
            "correctness_tail": check_text[-2000:],
            "benchmark_tail": (
                (bench.stdout + "\n" + bench.stderr)[-2000:] if bench else ""
            ),
        }
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        failures += int(not ok)
        print(
            f"{cid}: {'PASS' if ok else 'FAIL'} elapsed={result['elapsed_sec']}s "
            f"prior_ratio={ratio_to_prior}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
