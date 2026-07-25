#!/usr/bin/env python3
"""Show: (1) estimation model collapses kernels that differ only in
pipeline/wave (collision), (2) the simulation (Formocast) model can
distinguish them once the fine-grained tensile knobs are populated."""
import origami

hw = origami.get_hardware_for_device(0)
n_cu = int(hw.N_CU)

def dt(t):
    return origami.string_to_datatype(t)

def problem(m, n, k):
    p = origami.problem_t()
    p.size = origami.dim3_t(m, n, k)
    p.batch = 1
    p.a_transpose = origami.transpose_t.T
    p.b_transpose = origami.transpose_t.N
    p.a_dtype = dt("f8"); p.b_dtype = dt("f8")
    p.c_dtype = dt("bf16"); p.d_dtype = dt("bf16")
    p.mi_dtype = dt("f8")
    return p

def base_cfg(mt, mi, occ=2):
    c = origami.config_t()
    c.mt = origami.dim3_t(*mt)
    c.mi = origami.dim3_t(*mi)
    c.occupancy = occ
    return c

def with_sim(mt, mi, wave_m, wave_n, pipe_v, block_size=256, occ=2):
    c = base_cfg(mt, mi, occ)
    c.prediction_mode = origami.prediction_modes_t.simulation
    tp = origami.tensile_params_t()
    tp.wave_group_m = wave_m          # <- WAVE_MAP_M (MXdlPerWave)
    tp.wave_group_n = wave_n          # <- WAVE_MAP_N (NXdlPerWave)
    tp.wave_num = block_size // 64    # 256/64 = 4 waves
    tp.depth_u = mt[2]               # K-unroll ~ KPerBLOCK
    tp.global_split_u = 1
    tp.prefetch_global_read = 2 if pipe_v == 3 else 1   # v3 intrawave vs v1
    c.set_tensile_params(tp)
    return c

prob = problem(2048, 2048, 3072)

# Two aiter kernels that COLLIDE under estimation:
#   kid 2  : 64x128x128, MFMA 32x32, WAVE 1x2, pipeline v3
#   kid 15 : 64x128x128, MFMA 32x32, WAVE 2x1, pipeline v1
MT = (64, 128, 128); MI = (32, 32, 32)

print(f"device gfx950 N_CU={n_cu}   problem=(2048,2048,3072) fp8 TN\n")
print("ESTIMATION model (default; reads only mt/mi/occupancy/cache/grid):")
for name, wm, wn in [("kid2  (WAVE 1x2, v3)", 1, 2), ("kid15 (WAVE 2x1, v1)", 2, 1)]:
    lat = origami.compute_total_latency(prob, hw, base_cfg(MT, MI), n_cu)
    print(f"  {name}: {lat:12.1f} cycles")
print("  -> identical: estimation cannot see wave map / pipeline (COLLISION)\n")

print("SIMULATION model (Formocast; reads full tensile knobs):")
for name, wm, wn, pv in [("kid2  (WAVE 1x2, v3)", 1, 2, 3),
                         ("kid15 (WAVE 2x1, v1)", 2, 1, 1)]:
    try:
        lat = origami.compute_total_latency(prob, hw, with_sim(MT, MI, wm, wn, pv), n_cu)
        print(f"  {name}: {lat:12.1f} cycles")
    except Exception as e:
        print(f"  {name}: ERROR {type(e).__name__}: {e}")
print("  -> now differ (if finite): fine-grained knobs are consumed")
