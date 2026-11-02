# Structure Bridge MIta v3

Date: 2026-11-01

## Audit

Wall, opening, junction, Node, Edge, and GED are downstream of the structure
bridge. The previous typed bridge had only about 0.77M parameters and no global
attention:

| Version | Parameters | Structure branch |
|---|---:|---|
| typed v2 | 776,222 | 504,182 |
| MIta v3 | 2,903,184 | 2,631,144 |

The new candidate adds 2.13M parameters to the structure branch instead of the
shared room encoder. Its MIta memory alone is 645,906 parameters.

## Multi-Scale Routing

The local U-Net remains intact. A sparse global branch runs at the structure
bottleneck:

```text
room/structure context
-> 32x32 bottleneck
-> pool bottleneck, 64-scale, and 128-scale evidence
-> multi-scale MIta memory
-> sample-conditioned top-k budget
-> gated bottleneck residual
-> local 64/128 decoder
-> wall / opening / junction heads
```

This provides a direct path for global room layout, owner boundaries, wall
continuity, and opening-host relations to influence every structure head. It
does not replace the local decoder, which remains responsible for precise
boundaries.

## Budget

| Stage | Cases | Epochs | Batch | Exposures | Updates |
|---|---:|---:|---:|---:|---:|
| Screen | 512 | 12 | 8 | 6,144 | 768 |
| Promotion | 1,024 | 16 | 8 | 16,384 | 2,048 |
| Confirmation | 5,126 | 16 | 16 | 82,016 | 5,136 |

The previous three-epoch bridge run cannot be used as evidence against the new
capacity or routing branch.

## Required Controls

The full route must beat equal-parameter local convolution, static-budget MIta,
and no-multi-scale-memory controls. Wall precision and F1 are primary; opening,
junction, graph, validity, and watertightness are non-regression gates.

Route entropy, budget, expert fraction, and gate mean are emitted with every
validation pass.

## Files

- `src/floorflow/heads/structure_mita_router.py`
- `src/floorflow/heads/coarse_structure_refiner.py`
- `src/floorflow/heads/joint_control_repair.py`
- `src/floorflow/training/raster_structure_bridge_cli.py`
- `configs/structure_bridge_mita_v3.json`
