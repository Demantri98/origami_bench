import csv, os, sys
from collections import Counter

# Use the repo's tuned CSV that the user is inspecting.
TUNED = "/home/demantri/aiter/aiter/configs/a8w8_blockscale_tuned_gemm.csv"
SYNTH = "/home/demantri/origami_bench/datasets/a8w8_blockscale_gfx950_synth.csv"

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config

print("gfx:", get_gfx(), "cu_num:", get_cu_num())

shapes = []
with open(SYNTH) as f:
    for row in csv.DictReader(f):
        shapes.append((int(row["M"]), int(row["N"]), int(row["K"])))

tuned_hits = 0
fallback = 0
kern_counter = Counter()
lib_counter = Counter()
fallback_examples = []
tuned_examples = []

for (M, N, K) in shapes:
    cfg = get_CKGEMM_config(M, N, K, TUNED)
    if cfg is None:
        fallback += 1
        if len(fallback_examples) < 10:
            fallback_examples.append((M, N, K))
    else:
        tuned_hits += 1
        lib_counter[cfg["libtype"]] += 1
        kern_counter[cfg["kernelName"]] += 1
        if len(tuned_examples) < 5:
            tuned_examples.append((M, N, K, cfg["libtype"], cfg["kernelName"]))

print("\n==== REAL AITER DISPATCH RESULTS (get_CKGEMM_config) ====")
print("total synth shapes :", len(shapes))
print("tuned-config hits  :", tuned_hits)
print("fallback (backup CK):", fallback)
print("\nlibtype breakdown:", dict(lib_counter))
print("\nkernel breakdown:")
for k, v in kern_counter.most_common():
    print(f"  {v:5d}  {k}")
print("\nfallback examples:", fallback_examples)
print("\ntuned examples:")
for e in tuned_examples:
    print("  ", e)
