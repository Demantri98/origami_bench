import csv
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
from aiter.jit.core import AITER_CONFIGS

TUNED = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
DS = "/home/demantri/origami_bench/datasets/a8w8_blockscale_gfx950_novel_nk.csv"

rows = [(int(r["M"]), int(r["N"]), int(r["K"])) for r in csv.DictReader(open(DS))]
none_cnt = 0
non_none = []
for (M, N, K) in rows:
    cfg = get_CKGEMM_config(M, N, K, TUNED)
    if cfg is None:
        none_cnt += 1
    elif len(non_none) < 5:
        non_none.append((M, N, K, cfg.get("kernelName", "")))

print(f"config lookup (merged tuned file): {TUNED}")
print(f"get_CKGEMM_config == None (=> DEFAULT kernel) for {none_cnt}/{len(rows)} rows")
print(f"rows that resolve to a tuned config (should be 0): {len(rows) - none_cnt}")
if non_none:
    print("unexpected tuned matches:", non_none)
