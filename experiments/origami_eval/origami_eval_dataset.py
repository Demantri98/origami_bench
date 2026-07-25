#!/usr/bin/env python3
"""Generalized Origami-vs-measured eval for an arbitrary (M,N,K) dataset.

For each shape:
  * Origami estimation pick (with cache-hint fallback) — splitK=0.
  * Measure all 19 CK candidate kernels on-GPU @ splitK=0 -> oracle.
  * Production pick via aiter's REAL dispatch get_CKGEMM_config():
      - tuned row found (libtype=ck) -> that kernelId
      - no row / empty name       -> DEFAULT/backup kernel (kid 7)   [prod_is_default]
  * Default kernel (kid 7) is measured for every shape regardless.
Reports default-invocation rate + Origami/production/default vs the measured oracle.
"""
import argparse, csv, os, sys, statistics as st
from collections import defaultdict

import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
from aiter.jit.core import AITER_CONFIGS

sys.path.insert(0, "/home/demantri/origami_bench")
sys.path.insert(0, "/home/demantri/origami_bench/tools")
import origami
import ck_kernel_map
from common.families import get_family

TUNED = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
DEFAULT_KID = 7
BLOCK = (128, 128)
BUCKETS = [(0,16),(16,64),(64,256),(256,1024),(1024,4096),(4096,10**12)]
BLABEL = {(0,16):"≤16",(16,64):"16-64",(64,256):"64-256",(256,1024):"256-1024",
          (1024,4096):"1024-4096",(4096,10**12):">4096"}
FIELDS = ["M","N","K","o_kid","hint_mode","o_us","o_slow","oracle_kid","oracle_us",
          "prod_kid","prod_lib","prod_is_default","prod_us","prod_slow",
          "default_us","default_slow","o_vs_default","o_vs_prod","n_ok"]

def bof(m):
    for lo,hi in BUCKETS:
        if lo<m<=hi: return (lo,hi)

def gen(M,N,K,seed=0):
    torch.manual_seed(seed)
    bn,bk=BLOCK; sk=(K+bk-1)//bk; sn=(N+bn-1)//bn
    x=(torch.rand((M,K),dtype=dtypes.fp16,device="cuda")/10).to(dtypes.fp8)
    w=(torch.rand((N,K),dtype=dtypes.fp16,device="cuda")/10).to(dtypes.fp8)
    xs=torch.rand([M,sk],dtype=dtypes.fp32,device="cuda")
    ws=torch.rand([sn,sk],dtype=dtypes.fp32,device="cuda")
    out=torch.empty(M,N,dtype=dtypes.bf16,device="cuda")
    return x,w,xs,ws,out

def measure(kid,x,w,xs,ws,out,warmup,iters):
    try:
        _,us=run_perftest(aiter.gemm_a8w8_blockscale_tune,x,w,xs,ws,out,kid,0,
                          num_warmup=warmup,num_iters=iters,num_rotate_args=1,use_cuda_event=True)
        torch.cuda.synchronize(); return float(us)
    except Exception:
        return float("inf")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--warmup",type=int,default=3)
    ap.add_argument("--iters",type=int,default=10)
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args()

    shapes=[(int(r["M"]),int(r["N"]),int(r["K"])) for r in csv.DictReader(open(args.dataset))]
    if args.limit: shapes=shapes[:args.limit]

    fam=get_family("a8w8_blockscale")
    hw=origami.get_hardware_for_device(0); n_cu=int(hw.N_CU)
    base=ck_kernel_map.build_configs(origami,hw,fam,"",occupancy=2)
    ntb=ck_kernel_map.build_configs(origami,hw,fam,"",occupancy=2)
    for _,c in ntb: c.cache_hints_b=4
    nta=ck_kernel_map.build_configs(origami,hw,fam,"",occupancy=2)
    for _,c in nta: c.cache_hints_a=4
    def rank_fb(prob):
        for mode,cfgs in (("base",base),("nt_b",ntb),("nt_a",nta)):
            s=ck_kernel_map.rank_kernelids(origami,hw,prob,cfgs)
            if s: return s,mode
        return [],"none"
    ktable=ck_kernel_map.load_kernel_table(fam)
    name2kid={ki.name:kid for kid,ki in ktable.items()}

    print(f"Dataset {os.path.basename(args.dataset)}: {len(shapes)} shapes, gfx950 N_CU={n_cu}, "
          f"measure all 19 @sk0 (warmup={args.warmup} iters={args.iters})",flush=True)

    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    fh=open(args.out,"w",newline=""); wr=csv.DictWriter(fh,fieldnames=FIELDS); wr.writeheader(); fh.flush()

    per=[]
    for i,(M,N,K) in enumerate(shapes):
        prob=origami.problem_t(); prob.size=origami.dim3_t(M,N,K); prob.batch=1
        prob.a_transpose=origami.transpose_t.T; prob.b_transpose=origami.transpose_t.N
        f8=origami.string_to_datatype("f8"); bf=origami.string_to_datatype("bf16")
        prob.a_dtype=f8; prob.b_dtype=f8; prob.c_dtype=bf; prob.d_dtype=bf; prob.mi_dtype=f8
        scored,hint=rank_fb(prob); o_kid=scored[0][0] if scored else -1

        # production pick via real dispatch
        cfg=get_CKGEMM_config(M,N,K,TUNED)
        if cfg is None:
            prod_kid,prod_lib,prod_is_def=DEFAULT_KID,"default",True
        else:
            lib=cfg["libtype"]; nm=str(cfg.get("kernelName",""))
            if lib=="ck" and nm in name2kid:
                prod_kid,prod_lib,prod_is_def=name2kid[nm],"ck",False
            elif lib=="ck" and nm=="":
                prod_kid,prod_lib,prod_is_def=DEFAULT_KID,"default",True
            else:
                prod_kid,prod_lib,prod_is_def=None,lib,False

        x,w,xs,ws,out=gen(M,N,K)
        meas={}
        for kid,_ in base:
            u=measure(kid,x,w,xs,ws,out,args.warmup,args.iters)
            if u!=float("inf"): meas[kid]=u
        del x,w,xs,ws,out; torch.cuda.empty_cache()
        if not meas: continue
        oracle_kid=min(meas,key=meas.get); oracle=meas[oracle_kid]
        o_us=meas.get(o_kid,float("inf")); d_us=meas.get(DEFAULT_KID,float("inf"))
        prod_us=meas.get(prod_kid,float("inf")) if prod_kid is not None else float("inf")
        row=dict(M=M,N=N,K=K,o_kid=o_kid,hint_mode=hint,o_us=round(o_us,3),
                 o_slow=round(o_us/oracle,4) if o_us!=float("inf") else "",
                 oracle_kid=oracle_kid,oracle_us=round(oracle,3),
                 prod_kid=prod_kid if prod_kid is not None else "",prod_lib=prod_lib,
                 prod_is_default=int(prod_is_def),
                 prod_us=round(prod_us,3) if prod_us!=float("inf") else "",
                 prod_slow=round(prod_us/oracle,4) if prod_us!=float("inf") else "",
                 default_us=round(d_us,3) if d_us!=float("inf") else "",
                 default_slow=round(d_us/oracle,4) if d_us!=float("inf") else "",
                 o_vs_default=round(o_us/d_us,4) if (o_us!=float("inf") and d_us!=float("inf")) else "",
                 o_vs_prod=round(o_us/prod_us,4) if (o_us!=float("inf") and prod_us!=float("inf")) else "",
                 n_ok=len(meas))
        per.append(row); wr.writerow(row); fh.flush()
        if (i+1)%50==0: print(f"  ...{i+1}/{len(shapes)}",flush=True)
    fh.close()

    # ---- summary ----
    def S(v):
        v=[x for x in v if isinstance(x,(int,float)) and x!=float("inf")]
        if not v: return None
        s=sorted(v); return dict(mean=st.mean(v),med=st.median(v),p90=s[min(len(s)-1,int(0.9*len(s)))],mx=max(v))
    def F(d): return "n/a" if d is None else f"mean={d['mean']:.3f} med={d['med']:.3f} p90={d['p90']:.3f} max={d['mx']:.3f}"
    def num(r,k): 
        x=r[k]; return float(x) if x!="" else float("inf")

    dflt_rate=sum(r["prod_is_default"] for r in per)/len(per)*100
    print("\n"+"#"*96)
    print(f"SUMMARY {os.path.basename(args.dataset)}  ({len(per)} shapes, splitK=0)")
    print(f"  DEFAULT-KERNEL invoked by production (get_CKGEMM_config None/empty): "
          f"{sum(r['prod_is_default'] for r in per)}/{len(per)} = {dflt_rate:.1f}%")
    prodlibs=defaultdict(int)
    for r in per: prodlibs[r["prod_lib"]]+=1
    print(f"  production libtype breakdown: {dict(prodlibs)}")
    for b in BUCKETS:
        rs=[r for r in per if bof(r["M"])==b]
        if not rs: continue
        hit=sum(1 for r in rs if r["o_kid"]==r["oracle_kid"])/len(rs)*100
        print(f"\n  M {BLABEL[b]} ({len(rs)} shapes)  default_invoked={sum(r['prod_is_default'] for r in rs)}/{len(rs)}")
        print(f"    Origami vs oracle : hit@1={hit:5.1f}%  slowdown {F(S([num(r,'o_slow') for r in rs]))}")
        print(f"    Production vs orac :               slowdown {F(S([num(r,'prod_slow') for r in rs]))}")
        print(f"    Default  vs oracle :               slowdown {F(S([num(r,'default_slow') for r in rs]))}")
        print(f"    Origami vs Default : ratio        {F(S([num(r,'o_vs_default') for r in rs]))}  (<1 = Origami faster)")
    # overall
    hit=sum(1 for r in per if r["o_kid"]==r["oracle_kid"])/len(per)*100
    print("\n  "+"="*80)
    print(f"  OVERALL ({len(per)} shapes): Origami hit@1={hit:.1f}%")
    print(f"    Origami   vs oracle: {F(S([num(r,'o_slow') for r in per]))}")
    print(f"    Production vs oracle: {F(S([num(r,'prod_slow') for r in per]))}")
    print(f"    Default   vs oracle: {F(S([num(r,'default_slow') for r in per]))}")
    print(f"    Origami   vs Default: {F(S([num(r,'o_vs_default') for r in per]))}  (<1 = Origami faster)")
    print(f"    Saved -> {args.out}")

if __name__=="__main__":
    main()
