import sys
sys.path.insert(0,"/home/demantri/origami_bench"); sys.path.insert(0,"/home/demantri/origami_bench/tools")
import origami, ck_kernel_map
from common.families import get_family
fam=get_family("a8w8_blockscale")
hw=origami.get_hardware_for_device(0)
configs=ck_kernel_map.build_configs(origami,hw,fam,"",occupancy=2)
def rank(M,N,K):
    p=origami.problem_t(); p.size=origami.dim3_t(M,N,K); p.batch=1
    p.a_transpose=origami.transpose_t.T; p.b_transpose=origami.transpose_t.N
    f8=origami.string_to_datatype("f8"); bf=origami.string_to_datatype("bf16")
    p.a_dtype=f8;p.b_dtype=f8;p.c_dtype=bf;p.d_dtype=bf;p.mi_dtype=f8
    return ck_kernel_map.rank_kernelids(origami,hw,p,configs)
for M in [1,2,4,8,16,32,64,128]:
    s=rank(M,2048,3072)
    print(f"M={M:4d} -> {len(s):2d} feasible; top3={[(k,round(l,0)) for k,l in s[:3]]}")

# --- test hypothesis: small-M rejection is due to missing non-temporal cache hint ---
print("\nWith cache_hints_b=4 (non-temporal B) set on all configs:")
cfgs2=[]
for kid,c in configs:
    c.cache_hints_b=4
    cfgs2.append((kid,c))
def rank2(M,N,K,cfgs):
    p=origami.problem_t(); p.size=origami.dim3_t(M,N,K); p.batch=1
    p.a_transpose=origami.transpose_t.T; p.b_transpose=origami.transpose_t.N
    f8=origami.string_to_datatype("f8"); bf=origami.string_to_datatype("bf16")
    p.a_dtype=f8;p.b_dtype=f8;p.c_dtype=bf;p.d_dtype=bf;p.mi_dtype=f8
    return ck_kernel_map.rank_kernelids(origami,hw,p,cfgs)
for M in [1,8,16,32]:
    s=rank2(M,2048,3072,cfgs2)
    print(f"  M={M:4d} -> {len(s):2d} feasible; top3={[(k,round(l,0)) for k,l in s[:3]]}")
