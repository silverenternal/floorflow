# Official Baseline Feasibility v2

Status: executable anchors frozen; no matched FloorFlow SOTA row enabled yet.

## Why This Gate Exists

The current oracle-control result is strong evidence that FloorFlow can exploit
an explicit room program and coarse layout controls: CubiCasa test503 reaches
room macro IoU `0.9718`, with validity `1.0000` and conditional watertightness
`0.9595`. This is an upper-bound diagnostic. Its controls are target-derived,
so it cannot be entered as open reconstruction, RPLAN generation, free-text
editing, or a HouseMind comparison.

The open CubiCasa result remains the real capability baseline: room macro IoU
`0.06844`, wall F1 `0.29891`, opening F1 `0.31312`, Node F1 `0.53753`, edge
overlap `0.13927`, and GED `11.89861`. The gap is in discovering and decoding
the plan without target-derived layout controls, not in the final local
refinement stage.

## Executable Baselines

| Method | Track | Official execution | Public anchor | Local reproduction | Matched row |
|---|---|---:|---:|---:|---:|
| CE2EPlan | RPLAN boundary generation | yes | FID 1.08 | FID 1.127 | blocked |
| Raster2Seq | CubiCasa reconstruction | yes | RoomF1 0.887 | RoomF1 0.88527 | blocked |

CE2EPlan commit `774e670941eeda3e6677303ff79e8d517ae0d1c0` generated all
12,002 RPLAN test cases and completed the released post-processing. Its FID is
within the predeclared absolute tolerance of `0.1`; paper-style structural
ratios are within `0.02`. The repository has no license file, so its code,
weights, and data stay outside redistributable packages. The independent GED
is `5.2323`, not the paper's `3.48`; the exact official GED implementation is
not released and must not be treated as the same evaluator.

Raster2Seq commit `a6c4e27a68d11d7a459f6e4a2601fd887227dd1a` evaluated all
509 official CubiCasa cases. RoomF1 is `0.885268`, within the predeclared
absolute tolerance of `0.003`. Its local compatibility patch only updates
obsolete framework APIs and fixes CubiCasa id dtype/preprocessing issues; it
does not change model semantics.

Raster2Seq remains blocked from a matched identity table because its official
preprocessing creates 509 cases from source regions while FloorFlow's protocol
uses 503 SVG floors. Twelve source folders differ in cardinality, including two
project floors absent from the released annotations. No authoritative
region-to-floor mapping exists.

## Blocked Or Absent Methods

| Method | State | Reason | Allowed use |
|---|---|---|---|
| HouseMind | weights and data only | official code remains `Code Soon`; image tokenizer, serialization, constrained decoding, VQ-VAE decode, and post-processing are missing | token probe and public paper context only |
| TLC-Plan | assets only | repository contains only README and one image | paper context only |
| FMLM | absent | no verified executable repository, weights, and post-processing contract | paper context only |
| RenoLayout | absent | no verified executable repository, weights, and post-processing contract | paper context only |

HouseMind weights load on GPU 0 and expose 256 outline plus 256 room codes, but
the 16-case generation probe emits token traces rather than editable floorplans.
Those traces have no valid Delta-IoU, Delta-MSE, Node F1, edge overlap, or GED.

## Claim Decision

The two official pipelines are now fixed references, but neither is a matched
FloorFlow victory. SOTA claims remain disabled until FloorFlow is trained or
adapted to the identical RPLAN contract and to a shared CubiCasa case-unit
manifest.

The next gate is SOTA-02: define one canonical vector output and evaluator
contract so that future generation, reconstruction, editing, and external rows
cannot silently mix case units or metric implementations.
