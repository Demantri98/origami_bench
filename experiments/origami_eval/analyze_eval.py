import csv, statistics as st
from collections import Counter, defaultdict

TILE={0:"128x128x128",1:"128x64x128",2:"64x128x128",3:"64x64x128",4:"16x256x128",
5:"16x128x128",6:"16x64x128",7:"16x128x256",8:"16x64x256",9:"32x256x128",10:"32x128x128",
11:"32x64x128",12:"32x128x256",13:"32x64x256",14:"64x256x128",15:"64x128x128",16:"64x64x128",
17:"64x128x256",18:"64x64x256"}
# true row counts per bucket in the full tuned CSV (from earlier inspection)
BUCKET_COUNT={(0,16):25,(16,64):30,(64,256):120,(256,1024):143,(1024,4096):750,(4096,10**9):5562}
buckets=list(BUCKET_COUNT)
def bof(m):
    for lo,hi in buckets:
        if lo<m<=hi: return (lo,hi)

rows=[r for r in csv.DictReader(open("/home/demantri/origami_bench/results/estimation_vs_measured.csv"))]
for r in rows:
    for k in ("M","N","K","o_kid","oracle_kid","csv_kid"): r[k]=int(r[k])
    for k in ("o_slow","csv_slow","default_slow","o_us","oracle_us"): r[k]=float(r[k])

# production-weighted mean slowdown (weight each bucket's sampled mean by true count)
def prod_weighted(field):
    num=den=0
    for b in buckets:
        rs=[r for r in rows if bof(r["M"])==b and r[field]!=float("inf")]
        if not rs: continue
        num+=st.mean([r[field] for r in rs])*BUCKET_COUNT[b]; den+=BUCKET_COUNT[b]
    return num/den
print("PRODUCTION-WEIGHTED mean slowdown vs oracle (weighting buckets by real dataset frequency):")
print(f"  Origami  = {prod_weighted('o_slow'):.3f}x")
print(f"  CSV pick = {prod_weighted('csv_slow'):.3f}x")
print(f"  default7 = {prod_weighted('default_slow'):.3f}x")

print("\nWORST 12 Origami misses (o_slow):")
print(f"  {'M':>6} {'N':>5} {'K':>5} {'origami':>13} {'oracle':>13} {'slow':>6} {'oracleUs':>9}")
for r in sorted(rows,key=lambda r:-r["o_slow"])[:12]:
    print(f"  {r['M']:>6} {r['N']:>5} {r['K']:>5} {TILE[r['o_kid']]:>13} {TILE[r['oracle_kid']]:>13} "
          f"{r['o_slow']:>6.2f} {r['oracle_us']:>9.2f}")

# systematic bias: in mid-M (64..1024), what K does origami pick vs oracle?
mid=[r for r in rows if 64<r["M"]<=1024]
def kdist(kids): 
    c=Counter(TILE[k].split("x")[2] for k in kids); return dict(sorted(c.items()))
print(f"\nMID-M (64..1024, {len(mid)} shapes) K_block choice:")
print(f"  Origami picks K: {kdist([r['o_kid'] for r in mid])}")
print(f"  Oracle    is  K: {kdist([r['oracle_kid'] for r in mid])}")
print(f"  Origami picks tile: {dict(Counter(TILE[r['o_kid']] for r in mid).most_common(5))}")
print(f"  Oracle    is  tile: {dict(Counter(TILE[r['oracle_kid']] for r in mid).most_common(5))}")
