# Architecture Necessity Ablations

| Component | Arm A | Arm B | Matched evidence | Decision |
|---|---|---|---|---|
| Control gate | Raw local output outside exact 0.986009 | Gated outside exact 1.000000 | pixel gain +0.013991 [+0.012627, +0.015443]; room gain +0.016525 | retain |
| Watertight guard | Off 0.957875 | On 0.910044 | matched seed19, 2479 cases | do_not_retain |
| Graph-token / chip conditioning | Not part of retained floorplan route | N/A | architecture audit; separate FloorSet/chip track | not applicable |

The control gate preserves inside response at 0.9254 of the raw inside change.
