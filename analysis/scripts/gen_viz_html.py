#!/usr/bin/env python3
"""Emit a self-contained HTML dashboard (no external deps) with two SVG LINE
charts (one per benchmark run): X = N*K (log), Y = speedup, one line per M regime.
Reads /home/demantri/origami_bench/analysis/data/viz_line.json (produced by aggregate_line_viz.py)."""
import json, os

DATA = json.load(open("/home/demantri/origami_bench/analysis/data/viz_line.json"))
OUT_DIR = "/home/demantri/origami_bench/analysis/dashboard"
os.makedirs(OUT_DIR, exist_ok=True)

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Origami speedup - line graphs</title>
<style>
  :root{ --bg:#0d1117; --panel:#161b22; --stroke:#30363d; --text:#e6edf3; --muted:#9da7b3; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:28px;}
  h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:0 0 2px}
  .sub{color:var(--muted);max-width:940px;margin:0 0 20px}
  .cap{color:var(--muted);font-size:13px;margin:2px 0 6px}
  section{background:var(--panel);border:1px solid var(--stroke);border-radius:10px;
          padding:18px;margin:18px 0;max-width:980px}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0 2px;font-size:13px;color:var(--muted)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .sw{width:14px;height:3px;border-radius:2px;display:inline-block}
  svg{display:block;width:100%;height:auto}
  .axis{fill:var(--muted);font-size:11px}
  .axttl{fill:var(--muted);font-size:12px}
</style></head>
<body>
  <h1>Origami speedup - line graphs by M regime</h1>
  <p class="sub">X axis: N&middot;K (weight size, log scale). Y axis: speedup (&times;). One line per M regime
  (rows = activation tokens). Dashed line = 1.0&times; parity. All points are medians of on-GPU splitK=0
  measurements on gfx950. Hover any marker for the exact shape, value, and sample count.</p>
  <div id="root"></div>

<script>
const DATA = __DATA__;
const REGIMES = DATA.regimes;
const COLORS = { "<64":"#f85149", "64-256":"#d29922", "256-1024":"#3fb950",
                 "1024-4096":"#58a6ff", ">4096":"#bc8cff" };
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

function fmtNK(v){
  if(v>=1e9) return (v/1e9).toFixed(v>=1e10?0:1)+"B";
  if(v>=1e6) return (v/1e6).toFixed(v>=1e7?0:1)+"M";
  if(v>=1e3) return (v/1e3).toFixed(v>=1e4?0:1)+"K";
  return ""+v;
}
function xTicks(lo,hi){
  const t=[];
  for(let k=Math.floor(Math.log10(lo)); k<=Math.ceil(Math.log10(hi)); k++)
    for(const m of [1,2,5]){ const v=m*Math.pow(10,k); if(v>=lo*0.9 && v<=hi*1.1) t.push(v); }
  return t;
}
function lineChart(run){
  const W=920,H=380,padL=62,padR=22,padT=16,padB=54;
  // domains
  let xs=[], ys=[1.0];
  REGIMES.forEach(r=>(run.series[r]||[]).forEach(p=>{ xs.push(p.NK); ys.push(p.y); }));
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  const ymin=Math.min(...ys)*0.97, ymax=Math.max(...ys)*1.03;
  const lx=v=>Math.log10(v);
  const X=v=>padL+(W-padL-padR)*(lx(v)-lx(xmin))/((lx(xmax)-lx(xmin))||1);
  const Y=v=>padT+(H-padT-padB)*(1-(v-ymin)/((ymax-ymin)||1));
  let s=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMinYMin meet" role="img">`;
  // Y gridlines + ticks
  for(let i=0;i<=5;i++){ const v=ymin+(ymax-ymin)*i/5, y=Y(v);
    s+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="#30363d" stroke-width="1"/>`;
    s+=`<text class="axis" x="${padL-8}" y="${y+3}" text-anchor="end">${v.toFixed(2)}\u00d7</text>`;
  }
  // X ticks
  xTicks(xmin,xmax).forEach(v=>{ const x=X(v);
    s+=`<line x1="${x}" y1="${padT}" x2="${x}" y2="${H-padB}" stroke="#21262d" stroke-width="1"/>`;
    s+=`<text class="axis" x="${x}" y="${H-padB+16}" text-anchor="middle">${fmtNK(v)}</text>`;
  });
  // 1.0x reference
  if(1.0>=ymin && 1.0<=ymax){ const y1=Y(1.0);
    s+=`<line x1="${padL}" y1="${y1}" x2="${W-padR}" y2="${y1}" stroke="#8b949e" stroke-width="1.5" stroke-dasharray="5 4"/>`;
    s+=`<text class="axis" x="${W-padR}" y="${y1-5}" text-anchor="end" fill="#8b949e">1.0\u00d7 parity</text>`;
  }
  // axis titles
  s+=`<text class="axttl" x="${(padL+W-padR)/2}" y="${H-8}" text-anchor="middle">N\u00b7K (weight size, log scale)</text>`;
  s+=`<text class="axttl" transform="translate(14,${(padT+H-padB)/2}) rotate(-90)" text-anchor="middle">speedup (\u00d7)</text>`;
  // series
  REGIMES.forEach(r=>{
    const pts=(run.series[r]||[]).slice().sort((a,b)=>a.NK-b.NK), c=COLORS[r];
    if(pts.length>=2){
      const poly=pts.map(p=>`${X(p.NK).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
      s+=`<polyline points="${poly}" fill="none" stroke="${c}" stroke-width="2"/>`;
    }
    pts.forEach(p=>{
      s+=`<circle cx="${X(p.NK).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="3.4" fill="${c}">`+
         `<title>${esc(r)} regime | ${esc(p.nk)} (N\u00b7K=${p.NK}) | ${p.y.toFixed(2)}\u00d7 | n=${p.n}</title></circle>`;
    });
  });
  s+=`</svg>`;
  return s;
}
function legendHTML(run){
  return `<div class="legend">`+REGIMES.map(r=>{
    const has=(run.series[r]||[]).length;
    return `<span style="${has?'':'opacity:.4'}"><i class="sw" style="background:${COLORS[r]}"></i>M ${esc(r)}${has?'':' (no data)'}</span>`;
  }).join("")+`</div>`;
}
const root=document.getElementById("root");
DATA.runs.forEach(run=>{
  const sec=document.createElement("section");
  sec.innerHTML=`<h2>${run.title}</h2><div class="cap"><b>Y:</b> ${run.ylabel}. ${run.note}</div>`
    + legendHTML(run) + lineChart(run);
  root.appendChild(sec);
});
</script>
</body></html>
"""

html = HTML.replace("__DATA__", json.dumps(DATA))
path = os.path.join(OUT_DIR, "index.html")
open(path, "w").write(html)
print("wrote", path, os.path.getsize(path), "bytes")
