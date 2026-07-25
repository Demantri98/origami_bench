# Experiment artifacts

This directory contains the exploratory and GPU-measured work built around the
core `origami_bench` datasets and tools.

- `aiter_dispatch/` — aiter blockscale dispatch and default-kernel checks.
- `ck_template_sweep/` — all-19-candidate CK template sweeps and build trigger.
- `origami_eval/` — Origami analytical-ranking versus measured-kernel scripts.
- `roofline/` — gfx950 Omniperf/rocprof roofline pipeline, raw workloads, and plots.
- `kernelforge_ab/` — the matched KernelForge starting-template A/B pilot,
  including manifests, isolated git worktrees, campaign archives, rebenchmarks,
  paired outputs, and the final report.

Reusable source datasets remain in `../datasets/`, canonical measurements in
`../results/`, visualization tooling in `../analysis/`, and third-party tooling
in `../vendor/`.
