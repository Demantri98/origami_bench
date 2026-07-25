#!/usr/bin/env python3
"""fp8 roofline comparisons (Omniperf empirical ceilings, gfx950):
  - tuned shapes     : Origami vs exhaustive-tuned (oracle = best of 19 measured)
  - novel_nk (synth) : Origami vs default CK kernel (what aiter runs when untuned)
Origami is drawn on top so it stays visible where it coincides with the other."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

d = json.load(open("/home/demantri/origami_bench/experiments/roofline/roof_points.json"))
C = d["ceilings"]
fp8 = C["fp8_gflops"] / 1e3          # -> TFLOPS
bw = C["hbm_gbs"]                     # GB/s ; TFLOPS = AI * bw/1000
ridge = fp8 / (bw / 1000.0)

STY = {"origami": ("#2563eb", "o", "Origami pick"),
       "oracle":  ("#16a34a", "s", "Exhaustive tuned (oracle)"),
       "default": ("#dc2626", "^", "Default CK kernel")}

# (dataset, reference-variant-to-compare-against, panel title)
PANELS = [("tuned", "oracle", "tuned  ·  Origami vs exhaustive-tuned"),
          ("novel_nk", "default", "novel_nk (synthetic, untuned N,K)  ·  Origami vs default CK")]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
for ax, (ds, ref, title) in zip(axes, PANELS):
    pts = [p for p in d["points"] if p["dataset"] == ds and p["variant"] in ("origami", ref)]
    ai_all = [p["ai"] for p in pts] or [1]
    xlo, xhi = min(ai_all) / 2, max(max(ai_all) * 2, ridge * 1.5)
    ax.plot([xlo, ridge], [bw / 1000 * xlo, fp8], color="#111", lw=1.6)
    ax.plot([ridge, xhi], [fp8, fp8], color="#111", lw=1.6)
    ax.text(xhi, fp8 * 1.05, f"fp8 MFMA peak {fp8:.0f} TFLOPS", ha="right", fontsize=9)
    ax.text(xlo * 1.15, bw / 1000 * xlo * 1.15, f"HBM {bw/1000:.2f} TB/s", rotation=30, fontsize=9, color="#333")

    byshape = defaultdict(dict)
    for p in pts:
        byshape[(p["M"], p["N"], p["K"])][p["variant"]] = p
    # connectors show the Origami -> reference shift per shape
    for trip in byshape.values():
        if "origami" in trip and ref in trip:
            ax.plot([trip["origami"]["ai"], trip[ref]["ai"]],
                    [trip["origami"]["tflops"], trip[ref]["tflops"]], color="#bbb", lw=0.8, zorder=1)
    # reference first (large, filled), Origami last (small, on top) -> both visible when coincident
    rc, rm, rl = STY[ref]
    rp = [p for p in pts if p["variant"] == ref]
    ax.scatter([p["ai"] for p in rp], [p["tflops"] for p in rp], c=rc, marker=rm, s=140,
               edgecolors="white", linewidths=0.6, label=rl, zorder=2)
    oc, om, ol = STY["origami"]
    op = [p for p in pts if p["variant"] == "origami"]
    ax.scatter([p["ai"] for p in op], [p["tflops"] for p in op], c=oc, marker=om, s=48,
               edgecolors="white", linewidths=0.8, label=ol, zorder=4)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity  (FLOP / HBM byte)")
    ax.set_title(f"{title}   ({len(byshape)} shapes)")
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
axes[0].set_ylabel("Achieved performance  (TFLOPS, fp8)")
fig.suptitle("gfx950 fp8 roofline (Omniperf empirical ceilings) — small blue Origami markers sit on top",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/home/demantri/origami_bench/experiments/roofline/roofline_compare.png", dpi=130)
print("saved /home/demantri/origami_bench/experiments/roofline/roofline_compare.png")
