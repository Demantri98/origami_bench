#!/usr/bin/env python3
"""Seed editable candidate slot 0 from an existing CK candidate kernel.

The experiment driver always invokes kernelId 0.  Copying either the
Origami-selected candidate or default candidate 7 into slot 0 gives both arms
the same editable source location while preserving every other candidate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


INSTANCE_REL = Path(
    "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py"
)


def _load_candidates(path: Path):
    spec = importlib.util.spec_from_file_location("kf_ab_blockscale_instances", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.candidate_kernels_dict


def _candidate_span(lines: list[str]) -> tuple[int, int]:
    start = next(i for i, line in enumerate(lines) if line.startswith("candidate_kernels_dict = {"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return start, end


def _row_index(lines: list[str], start: int, end: int, kernel_id: int) -> int:
    pattern = re.compile(rf"^\s*{kernel_id}:\s*KernelInstance\(")
    matches = [i for i in range(start + 1, end) if pattern.match(lines[i])]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one candidate row for kernelId {kernel_id}, found {len(matches)}"
        )
    return matches[0]


def _instance_dict(instance) -> dict:
    return {
        field: getattr(instance, field)
        for field in instance.__dataclass_fields__
    } | {"name": instance.name}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--source-kid", type=int, required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    worktree = Path(args.worktree).resolve()
    path = worktree / INSTANCE_REL
    before = path.read_text().splitlines()
    start, end = _candidate_span(before)
    source_idx = _row_index(before, start, end, args.source_kid)
    target_idx = _row_index(before, start, end, 0)

    # Keep the target mapping key fixed at 0, copy only the constructor.
    constructor = before[source_idx].split("KernelInstance(", 1)[1]
    replacement = "    0:   KernelInstance(" + constructor
    after = list(before)
    after[target_idx] = replacement
    path.write_text("\n".join(after) + "\n")

    # Verify no line outside candidate slot 0 changed.
    changed = [i + 1 for i, (a, b) in enumerate(zip(before, after)) if a != b]
    if changed not in ([], [target_idx + 1]):
        raise RuntimeError(f"unexpected changed lines: {changed}")

    candidates = _load_candidates(path)
    seeded = candidates[0]
    source = candidates[args.source_kid]
    if _instance_dict(seeded) != _instance_dict(source):
        raise RuntimeError("seeded slot 0 does not match requested source candidate")

    metadata = {
        "worktree": str(worktree),
        "source_kernel_id": args.source_kid,
        "editable_kernel_id": 0,
        "instance_file": str(path),
        "changed_lines": changed,
        "seeded_template": _instance_dict(seeded),
    }
    meta_path = Path(args.metadata)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
