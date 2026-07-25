#!/usr/bin/env python3
"""Run prepared KernelForge campaigns sequentially inside origami-dev."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
CONTAINER = "origami-dev"
CONTAINER_USER = "15724:15724"
INSTANCE_REL = "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py"


def campaign_id(campaign: dict) -> str:
    return f"{campaign['shape_id']}-{campaign['arm_id']}"


def shell_quote(value) -> str:
    return shlex.quote(str(value))


def command_for(campaign: dict, max_iters: int, max_hours: float, profiling: bool) -> list[str]:
    cid = campaign_id(campaign)
    run_dir = HERE / "runs" / cid
    meta = json.loads((run_dir / "campaign.json").read_text())
    worktree = Path(meta["worktree"])
    shape = {"M": campaign["M"], "N": campaign["N"], "K": campaign["K"]}
    shapes = {"primary": shape, "minimal": shape, "validation": [shape]}
    profile_flag = "--profiling" if profiling else "--no-profiling"
    payload = f"""
set -e
export KERNEL_AGENTS_MODEL="${{CLAUDE_MODEL:-${{KERNEL_AGENTS_MODEL:-}}}}"
export FORGE_CLAUDE_BIN="/usr/local/lib/python3.12/dist-packages/claude_agent_sdk/_bundled/claude"
export PATH="/home/demantri/origami_bench/vendor/rocprofiler-compute/src:$PATH"
export PYTHONPATH="/home/demantri/KernelForge/src:{worktree}:$PYTHONPATH"
export AITER_REBUILD=2
export KF_AITER_ROOT={shell_quote(worktree)}
export KF_PRIMARY_SHAPE={shell_quote(f"M={campaign['M']},N={campaign['N']},K={campaign['K']}")}
export GIT_AUTHOR_NAME="KernelForge Pilot"
export GIT_AUTHOR_EMAIL="kernel-forge-pilot@local"
export GIT_COMMITTER_NAME="KernelForge Pilot"
export GIT_COMMITTER_EMAIL="kernel-forge-pilot@local"
cd /home/demantri/KernelForge
python3 -m kernel_agents.cli forge-loop \
  --kernel {shell_quote(worktree / INSTANCE_REL)} \
  --driver {shell_quote(worktree / "forge_driver.py")} \
  --workspace {shell_quote(worktree)} \
  --shapes-json {shell_quote(json.dumps(shapes, separators=(",", ":")))} \
  --snr-threshold 30 \
  --max-iters {max_iters} \
  --max-hours {max_hours} \
  --git-branch {shell_quote(meta["branch"])} \
  --gpu-target gfx950 \
  --fellow ck-fellow \
  --program-md-file {shell_quote(run_dir / "program.md")} \
  --experiments-dir {shell_quote(run_dir / ("experiments" if max_iters > 1 else "smoke_experiments"))} \
  --result-json {shell_quote(run_dir / ("result.json" if max_iters > 1 else "smoke_result.json"))} \
  --supervisor-backend claude \
  --profile-timeout-sec 1800 \
  {profile_flag} \
  --source-files {shell_quote(worktree / INSTANCE_REL)} \
  --target-functions candidate_kernels_dict \
  --framework-version 967a03ac19625163e8f9c4484a5fde364dfa87b8 \
  --workload-key {shell_quote(cid)}
"""
    inner = (
        "exec dotenv -f /home/demantri/Hyperloom/.env run -- "
        f"bash -lc {shell_quote(payload)}"
    )
    return [
        "docker",
        "exec",
        "-u",
        CONTAINER_USER,
        "-e",
        "HOME=/home/demantri",
        "-e",
        "USER=demantri",
        "-e",
        "LOGNAME=demantri",
        "-e",
        "PYTHONUNBUFFERED=1",
        CONTAINER,
        "bash",
        "-lc",
        inner,
    ]


def run_one(campaign: dict, max_iters: int, max_hours: float, profiling: bool) -> int:
    cid = campaign_id(campaign)
    run_dir = HERE / "runs" / cid
    log_path = run_dir / ("stdout.log" if max_iters > 1 else "smoke_stdout.log")
    status_path = run_dir / ("status.json" if max_iters > 1 else "smoke_status.json")
    cmd = command_for(campaign, max_iters, max_hours, profiling)
    started = time.time()
    print(f"\n===== {cid} order={campaign['run_order']} iters={max_iters} =====", flush=True)
    print("+ docker exec ... kernel_agents.cli forge-loop", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(f"[{cid}] {line}")
            sys.stdout.flush()
        rc = process.wait()
    status = {
        "campaign_id": cid,
        "returncode": rc,
        "max_iterations": max_iters,
        "max_hours": max_hours,
        "profiling": profiling,
        "elapsed_sec": round(time.time() - started, 3),
        "log": str(log_path),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"===== {cid} rc={rc} elapsed={status['elapsed_sec']}s =====", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated campaign IDs")
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--profiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="rerun completed result files")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    selected = {value.strip() for value in args.only.split(",") if value.strip()}
    campaigns = sorted(manifest["campaigns"], key=lambda item: item["run_order"])
    failures = 0
    for campaign in campaigns:
        cid = campaign_id(campaign)
        if selected and cid not in selected:
            continue
        run_dir = HERE / "runs" / cid
        if not (run_dir / "campaign.json").exists():
            raise RuntimeError(f"{cid} is not prepared")
        result_name = "result.json" if args.max_iters > 1 else "smoke_result.json"
        if (run_dir / result_name).exists() and not args.force:
            print(f"{cid}: result exists, skipping")
            continue
        rc = run_one(campaign, args.max_iters, args.max_hours, args.profiling)
        failures += int(rc != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
