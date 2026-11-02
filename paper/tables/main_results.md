# Main result table

The final candidate is deterministic and target-free, uses one sample per
condition, and applies the learned recursion policy only when it predicts that
direct decoding is insufficient.

| Dataset | Decoder | Split | Room IoU diagnostic | Instance IoU | Type accuracy | Room polygon F1 | Balanced CAD |
|---|---|---|---:|---:|---:|---:|---:|
| CubiCasa | room segmenter, direct | test503 | 0.17131 | 0.51706 | 0.42372 | 0.16995 | 0.31638 |
| CubiCasa | learned recursion, direct fallback | test503 | 0.17018 | 0.52363 | 0.43416 | 0.17346 | 0.33309 |
| CubiCasa | + MiTA graph, gated geometry/semantics | test503 | 0.16752 | 0.51673 | 0.42750 | 0.17817 | 0.33198 |
| CubiCasa | retrieval-free prior + learned gate + dynamic furniture queries | test503 | 0.16949 | 0.51775 | 0.43194 | 0.16075 | 0.33013 |

The gated graph candidate was selected on val64: Room IoU stayed within the
`0.002` guard while instance geometry rose from `0.50584` to `0.52701`,
polygon F1 from `0.16465` to `0.17220`, and balanced CAD from `0.31986` to
`0.32257`. On the frozen test503, the gains did not transfer: Room IoU and
instance geometry were `0.00266` and `0.00690` below the preceding candidate,
while polygon F1 and furniture containment improved. This is reported as a
generalization failure, not hidden by re-selecting on test.

The final retrieval-free candidate removes train-bank lookup from the main
path, uses the learned room-graph trust gate (`0.35`), dynamic furniture
queries, and eight style adapters with style 0 acting as the unmodified base.
It keeps Room IoU near the original decoder while adding opening and instance
gains. The style-enabled model reports style separation `0.03993` and
worst-style Room IoU `0.11488`.

## Complete CAD metric suite

| Family | Metric | Test503 value |
|---|---|---:|
| Pixel semantics | Room macro IoU diagnostic | 0.16949 |
| Pixel semantics | Room support IoU | 0.74833 |
| Pixel semantics | Boundary F1 | 0.27002 |
| Vector geometry | Room polygon / face F1 | 0.16075 / 0.24153 |
| Vector geometry | Wall F1 | 0.41955 |
| Topology | Validity | 1.00000 |
| Topology | Conditional watertight | 0.98649 |
| Topology | Unconditional watertight | 0.92247 |
| Spatial constraints | Room target recall | 0.99602 |
| Spatial constraints | Furniture-room containment | 0.81113 |
| Spatial constraints | Outside-footprint pixels | 0 |
| Instances | Geometric IoU / room F1 / type accuracy | 0.51775 / 0.15983 / 0.43194 |
| Objects | Opening / furniture IoU | 0.18838 / 0.06899 |
| Summary | Balanced CAD secondary | 0.33013 |

The conditional-generation hard gate passes:

| Requirement | Value | Threshold |
|---|---:|---:|
| FP-Eval validity | 1.00000 | 1.0 |
| Conditional watertight | 0.98649 | 0.95 |
| Room target recall | 0.99602 | 0.95 |
| Outside-footprint pixels | 0 | 0 |

CubiCasa ground-truth room occupancy covers only `74.83%` of footprint pixels
on average (maximum `93.65%`). The prior `room_coverage >= 0.95` floor was
therefore replaced by target-relative room recall.

## Learned recursion audit

| Predicted action / depth | Cases | Room IoU | Instance IoU | Type accuracy | Room polygon F1 |
|---|---:|---:|---:|---:|---:|
| Direct / depth 0 | 165 | 0.14756 | 0.43038 | 0.34909 | 0.13123 |
| Recursive / depth 1 | 161 | 0.17951 | 0.52251 | 0.44741 | 0.17424 |
| Split / depth 3 | 177 | 0.18083 | 0.59488 | 0.49510 | 0.17601 |

The learned gate uses footprint-only geometry at inference. Ground-truth room
statistics are used only to generate train labels and validation diagnostics.
