import sys
import torch
import aiter
from aiter import dtypes
sys.path.insert(0, "/home/demantri/origami_bench")
sys.path.insert(0, "/home/demantri/origami_bench/tools")
sys.path.insert(0, "/home/demantri/aiter/csrc/ck_gemm_a8w8_blockscale")
import origami
import ck_kernel_map
from common.families import get_family
from gemm_a8w8_blockscale_instance import candidate_kernels_dict as CK

TILE = {kid: f"{k.MPerBLOCK}x{k.NPerBLOCK}x{k.KPerBLOCK}" for kid, k in CK.items()}

def gen(M, N, K):
    bn, bk = 128, 128
    sk = (K + bk - 1)//bk; sn = (N + bn - 1)//bn
    x = (torch.rand((M, K), dtype=dtypes.fp16, device="cuda")/10).to(dtypes.fp8)
    w = (torch.rand((N, K), dtype=dtypes.fp16, device="cuda")/10).to(dtypes.fp8)
    xs = torch.rand([M, sk], dtype=dtypes.fp32, device="cuda")
    ws = torch.rand([sn, sk], dtype=dtypes.fp32, device="cuda")
    out = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
    return x, w, xs, ws, out

fam = get_family("a8w8_blockscale")
hw = origami.get_hardware_for_device(0)

def rank(M, N, K):
    base = ck_kernel_map.build_configs(origami, hw, fam, "", 2)
    for mode in ("base", "ntb"):
        cfgs = base
        if mode == "ntb":
            cfgs = ck_kernel_map.build_configs(origami, hw, fam, "", 2)
            for _, c in cfgs: c.cache_hints_b = 4
        p = origami.problem_t(); p.size = origami.dim3_t(M, N, K); p.batch = 1
        p.a_transpose = origami.transpose_t.T; p.b_transpose = origami.transpose_t.N
        f8 = origami.string_to_datatype("f8"); bf = origami.string_to_datatype("bf16")
        p.a_dtype = f8; p.b_dtype = f8; p.c_dtype = bf; p.d_dtype = bf; p.mi_dtype = f8
        s = ck_kernel_map.rank_kernelids(origami, hw, p, cfgs)
        if s:
            return s, mode
    return [], "none"

for (M, N, K) in [(8192, 64, 128), (2048, 64, 128), (256, 64, 128)]:
    x, w, xs, ws, out = gen(M, N, K)
    runnable, failed = [], []
    for kid in range(len(CK)):
        try:
            aiter.gemm_a8w8_blockscale_tune(x, w, xs, ws, out, kid, 0)
            torch.cuda.synchronize(); runnable.append(kid)
        except Exception as e:
            failed.append((kid, str(e).splitlines()[-1][:70]))
    del x, w, xs, ws, out; torch.cuda.empty_cache()
    s, mode = rank(M, N, K)
    pick = s[0][0] if s else None
    print(f"\n(M,N,K)=({M},{N},{K})")
    print(f"  CK-runnable kids ({len(runnable)}): {[(k, TILE[k]) for k in runnable]}")
    print(f"  CK-REJECTED kids ({len(failed)}): {[(k, TILE[k]) for k, _ in failed]}")
    if failed:
        print(f"    reject reason (kid {failed[0][0]}): {failed[0][1]}")
    print(f"  Origami top5 (hint={mode}): {[(k, TILE[k]) for k, _ in s[:5]]}")
    print(f"  Origami PICK = kid {pick} [{TILE.get(pick)}]  -> {'RUNNABLE' if pick in runnable else 'REJECTED by CK'}")
