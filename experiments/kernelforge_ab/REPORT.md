# KernelForge starting-template A/B pilot

Completed campaign artifacts: 12/12
Complete paired shapes: 6/6

## Paired outcomes

| Shape | M,N,K | Prior start advantage | Campaign final O/D | Same-session rebench O/D | Rebench winner | Same final template |
|---|---|---:|---:|---:|---|---|
| s01 | 10,4352,2304 | 1.128x | 1.0065 | 0.9631 | Origami start | False |
| s02 | 55,2112,768 | 1.014x | 0.9931 | 1.0153 | Default start | False |
| s03 | 86,20480,2048 | 1.221x | 0.9153 | 0.9174 | Origami start | False |
| s04 | 887,4352,2304 | 2.058x | 0.9925 | 1.0030 | Default start | True |
| s05 | 1741,4352,2304 | 2.478x | 1.0000 | 1.0097 | Default start | True |
| s06 | 19048,2112,768 | 2.567x | 1.0644 | 1.0091 | Default start | False |

## Pilot summary

- Median final Origami-start/default-start ratio: 0.9966.
- Origami-start final wins: 3/6.
- Same-session rebench median Origami/default ratio: 1.0061.
- Same-session Origami-start wins: 2/6.
- Applying KernelForge's 2% noise floor: 2 meaningful Origami-start wins, 0 meaningful default-start wins, and 4 ties.
- Both arms converged to the exact same final template on 2/6 shapes.
- Pilot conclusion: a stronger Origami baseline did not reliably produce a faster final kernel; the effect was meaningful on two shapes and erased by convergence/noise on four.
- This is a single-campaign-per-arm pilot and is descriptive, not a statistical test.
- A result below 1.0 means the campaign initialized from Origami ended faster.
