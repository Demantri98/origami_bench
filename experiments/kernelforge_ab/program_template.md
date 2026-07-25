# CK a8w8 blockscale template optimization

Optimize candidate slot `candidate_kernels_dict[0]` in:

`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py`

for the single target shape:

`M={M}, N={N}, K={K}`, FP8 inputs, FP32 block scales, BF16 output, gfx950
with 256 CUs and splitK=0.

## Objective

Minimize the driver-reported median kernel `wall_ms` while maintaining SNR of
at least 30 dB.  The baseline currently in slot 0 is intentionally the only
starting point supplied for this campaign.

## Allowed edit

Edit only the `KernelInstance(...)` constructor on the
`candidate_kernels_dict[0]` row.  Treat the other candidate rows, the default
dictionary, code generation, common CK implementation, driver, and tests as
immutable.  Do not change the candidate key (`0`).

The tunable fields are CK template parameters such as:

- `MPerBLOCK`, `NPerBLOCK`, `KPerBLOCK`
- `AK1`, `BK1`
- `MPerXDL`, `NPerXDL`
- wave mapping
- A/B block-transfer clusters
- CShuffle transfer configuration
- pipeline scheduler/version

Keep:

- `BLOCK_SIZE=256`
- scale blocks fixed at `1 × 128 × 128`
- a valid CK template combination that compiles and passes the driver

Use measured evidence.  Make one coherent template change at a time, let the
validation and benchmark gates decide, and do not infer success from compilation
alone.  Regressions, unsupported configurations, and numerical failures must be
reverted.
