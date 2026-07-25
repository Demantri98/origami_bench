#!/usr/bin/env python3
"""Create isolated, opaquely named aiter worktrees for all pilot campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
AITER = Path("/home/demantri/aiter")
WORKTREES = HERE / "worktrees"
RUNS = HERE / "runs"
INSTANCE_REL = Path(
    "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py"
)
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "KernelForge Pilot",
    "GIT_AUTHOR_EMAIL": "kernel-forge-pilot@local",
    "GIT_COMMITTER_NAME": "KernelForge Pilot",
    "GIT_COMMITTER_EMAIL": "kernel-forge-pilot@local",
}


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd, cwd=cwd, env=GIT_ENV, check=check, text=True, capture_output=True
    )


def campaign_id(campaign: dict) -> str:
    return f"{campaign['shape_id']}-{campaign['arm_id']}"


def guard_hash(path: Path) -> str:
    """Hash the candidate file excluding the one editable slot-0 row."""
    lines = path.read_text().splitlines()
    guarded = [
        line
        for line in lines
        if not line.lstrip().startswith("0:") or "KernelInstance(" not in line
    ]
    return hashlib.sha256(("\n".join(guarded) + "\n").encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove existing pilot worktrees/branches before recreating them",
    )
    parser.add_argument(
        "--only",
        default="",
        help="optional comma-separated campaign IDs (e.g. s01-u,s01-v)",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    base = manifest["base_aiter_commit"]
    selected = {value.strip() for value in args.only.split(",") if value.strip()}
    campaigns = [
        item
        for item in manifest["campaigns"]
        if not selected or campaign_id(item) in selected
    ]

    status = run(["git", "status", "--porcelain"], cwd=AITER).stdout.strip()
    if status:
        raise RuntimeError(f"main aiter worktree is not clean:\n{status}")
    actual = run(["git", "rev-parse", "HEAD"], cwd=AITER).stdout.strip()
    if actual != base:
        raise RuntimeError(f"aiter HEAD {actual} != manifest base {base}")

    WORKTREES.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    for campaign in campaigns:
        cid = campaign_id(campaign)
        branch = f"kf-ab-{cid}"
        worktree = WORKTREES / campaign["shape_id"] / campaign["arm_id"]
        run_dir = RUNS / cid
        if worktree.exists():
            if not args.force:
                print(f"{cid}: already prepared at {worktree}")
                continue
            run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=AITER,
                check=False,
            )
        if args.force:
            run(["git", "branch", "-D", branch], cwd=AITER, check=False)

        worktree.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        run(["git", "worktree", "add", "-b", branch, str(worktree), base], cwd=AITER)

        shutil.copy2(HERE / "forge_driver.py", worktree / "forge_driver.py")
        metadata_path = run_dir / "seed_metadata.json"
        seed = run(
            [
                sys.executable,
                str(HERE / "seed_template.py"),
                "--worktree",
                str(worktree),
                "--source-kid",
                str(campaign["source_kernel_id"]),
                "--metadata",
                str(metadata_path),
            ]
        )
        (run_dir / "seed_stdout.json").write_text(seed.stdout)

        # Neutral, shape-specific program: no arm provenance is disclosed.
        program = (HERE / "program_template.md").read_text().format(
            M=campaign["M"], N=campaign["N"], K=campaign["K"]
        )
        (run_dir / "program.md").write_text(program)

        campaign_meta = {
            **campaign,
            "campaign_id": cid,
            "branch": branch,
            "worktree": str(worktree),
            "run_dir": str(run_dir),
            "source_guard_sha256": guard_hash(worktree / INSTANCE_REL),
        }
        (run_dir / "campaign.json").write_text(
            json.dumps(campaign_meta, indent=2) + "\n"
        )

        run(["git", "add", str(INSTANCE_REL)], cwd=worktree)
        run(
            [
                "git",
                "commit",
                "--allow-empty",
                "-m",
                f"seed editable CK template for {campaign['shape_id']}/{campaign['arm_id']}",
            ],
            cwd=worktree,
        )
        print(f"{cid}: prepared {worktree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
