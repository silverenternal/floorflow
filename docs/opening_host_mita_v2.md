# Opening Host MIta v2

Date: 2026-11-01

## Audit

Opening is the weakest physical CAD layer. The existing `OpeningHostRouter`
used one dense opening-to-wall cross-attention, followed by a learned host
bilinear score and a geometric center-distance prior.

| Version | Parameters | Global host route |
|---|---:|---|
| dense router | about 66k | no |
| MIta v2 | 142,977 | 37,248 parameters |

The previous 64/32 eight-epoch opening-router screen was too small to support a
mechanism conclusion and did not beat detector fallback.

## Hybrid Host Routing

The router now keeps dense opening-to-wall attention and adds a sparse MIta
route over selected wall hosts:

```text
opening query
-> dense wall cross-attention
-> MIta wall-host routing
-> sample-conditioned top-k wall budget
-> gated host-context residual
-> typed host score + interval + type heads
```

The geometric distance prior remains unchanged. MIta changes the learned host
evidence, not the final physical legality check.

## Budget

| Stage | Cases | Epochs | Batch | Exposures | Updates |
|---|---:|---:|---:|---:|---:|
| Screen | 256 | 16 | 4 | 4,096 | 1,024 |
| Promotion | 1,024 | 20 | 4 | 20,480 | 5,120 |
| Confirmation | 5,126 | 16 | 8 | 82,016 | 10,256 |

The screen replaces the previous 64-case/8-epoch probe.

## Controls

The hybrid route must beat dense routing, static top-k, no-geometric-prior
ablation, and shuffled-opening-token controls. Opening F1 and host accuracy are
primary; center/width error, type, validity, and watertightness are
non-regression gates.

## Files

- `src/floorflow/heads/opening_host_router.py`
- `scripts/train_opening_host_router_v1.py`
- `configs/opening_host_mita_v2.json`
