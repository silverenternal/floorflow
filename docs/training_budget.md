# Current training budget

## Scale

| Item | Current |
| --- | ---: |
| Source cases, train / val / test | 4,199 / 399 / 399 |
| Floor samples at 256 px | 5,126 / 491 / 503 |
| Pixels per train epoch | 335,937,536 |
| Model parameters | 5,396,132 |
| Main-objective sample exposures | 50,742 |
| Equivalent full train epochs | 9.90 |
| Main-objective optimizer updates | 3,465 |
| Frozen-backbone object-head training | 5 epochs |

The latest full-data stage itself used six epochs at batch size 16, followed by
three lower-LR epochs. Earlier 512/1024 curriculum stages account for the
remaining effective exposure. This is small relative to typical semantic
segmentation or generative training runs, which commonly use tens to hundreds
of epochs.

## Long-tail coverage

The rarest room classes in train are Elevated (53 floors), Office (101), Sauna
(110), Alcove (127), Hall (129), the grouped rare-room class (188), CarPort
(233), and Den (243). Their test support is only 6–20 floors. Macro IoU over 24
classes therefore has high variance and is strongly affected by very narrow
data support.

Machine-readable details:

```text
outputs/reports/training_budget.json
```

## Is low IoU only a budget problem?

Partially, but not only:

- Training scale is genuinely modest, and rare-class supervision is small.
- More full-data epochs reached a plateau rather than continuing to improve.
- A ResNet18-FPN with more encoder parameters did not beat the 5.4M model
  (0.0849 vs about 0.10 on the 191-case diagnostic).
- A train-only contour memory prior gives a held-out improvement whose paired
  uncertainty estimate must accompany the claim, showing that the model is not
  fully exploiting available training layouts.
- Room types are under-identified from the exterior contour alone, so the task
  also needs a conditional generative/instance representation, not only more
  pixels or parameters.

## Next budget plan

1. Use synthetic floorplan pretraining from fpvec-lab/fpgen, then fine-tune on
   CubiCasa5K.
2. Once loss, representation, and decoding are frozen, train the main model for
   30–50 full-data epochs with per-head curves.
3. Train 384/512 resolution selectively for room/wall local structure rather
   than scaling the entire flow model blindly.
4. Require every budget increase to report full 503-case deltas and paired
   bootstrap before being retained.

## Current budget-gated decisions

The current predicted-control work uses a pre-execution review followed by a
short matched screen. A run is not rejected merely because it has not converged;
it is also not extended merely because more epochs remain.

| Experiment | Budget | Decision |
|---|---|---|
| Native room resolution | 256 vs 384, then 8/24/32/40 epochs | Keep 384px/32ep; 40ep regressed |
| 512px room identity | Matched 8ep | Reject: instance F1 and count regressed end to end |
| Direct-GT instance query | 8ep, then 10ep plateau | Keep as a mechanism; full-val topology regression blocks global use |
| Joint encoder update | 8ep at 2e-4 then 2e-5 | Reject: high rate collapsed, low rate remained below frozen encoder |
| Learned dense/query gate | 256 train cases, 8 gate epochs | Reject after safety-only ablation shows no reliable incremental benefit |
| Query routing + safety fallback | Full val491, deterministic inference | Retain: count, graph, and GED improve without structure regression |
| Coarse structure bridge | 3 -> 8 full-data epochs | Reject: wall/opening plateau and no end-to-end gain |
| Layered wall/opening readout | Zero-training cal64 probe before any training | Active: avoid spending epochs on a representation that cannot survive CAD decoding |
| Wall-centerline fusion | Zero-training cal64 grid | Wall precision improves; opening regresses, so only the wall layer survives |
| Structured opening / interval fusion | Zero-training cal64 probe | Reject threshold-only opening readout; opening F1 improves by at most 0.00127 |
| Frozen-feature opening detector | 8 epochs or plateau, train256/cal64, extend to 24 if improving | Active: replace the one-interval target, which collapses repeated openings present on a wall in 53/64 calibration plans |
| Dedicated opening detector scaling | 256 x 24, 1024 x 8, full5126 x 8 | 1024 x 8 is selected: oracle-host opening F1 0.3052 and best end-to-end 0.2650; full5126 does not improve end-to-end |
| Wall-band weighted opening objective | Matched train1024/cal64, 8 epochs | Reject: oracle-host opening F1 0.3007 versus 0.3052 for the unweighted control |
| Room-query full-resolution refinement | Matched train256/cal64, 8 epochs | Reject: Room IoU 0.33495->0.33516 but instance F1 0.27306->0.26680 and GED 9.75->9.8281 |
| Flat typed/dense all-channel heads | 256/64, 8 epochs | Reject as endpoint: Room IoU 0.0575/0.0524; loses CAD stage handoff |
| Staged room + typed channel graph | 256/64, 8 epochs | Room stage stabilized at 0.308; wall/opening/furniture still require dedicated stage heads |
| CAD action graph + stage-family loss | 256/64, 8 epochs | Room IoU stable at 0.308; opening F1 rose to 0.099; wall/furniture still limited by shared channel logits |
| Explicit CAD layer contract | 256/64, 8 epochs | Room IoU 0.3121, wall F1 0.5425, furniture F1 0.0971; opening remains 0.0528 and instance F1 regresses |
| CAD layer constraints | 256/64, 8 epochs | Room 0.3081, instance F1 0.2198, wall F1 0.5397, opening F1 0.2192, furniture F1 0.0981; plan vectorization remains the bottleneck |
| CAD vector layer readout | deterministic skeleton/segment vectorizer | Vector wall F1 0.3279, opening F1 0.0618, room F1 0.2248; direct raster-to-vector readout is not sufficient |
| CAD vector + room-boundary scaffold | deterministic skeleton/segment readout | Vector wall F1 improves to 0.3392 and opening F1 to 0.1170; room F1 stays 0.2248 |
| Learned vector topology scaffold | 256/64, 8 epochs | Vector wall F1 0.4297, opening F1 0.4145; next gate is end-to-end vector plan composition |
| Vector scaffold full-plan readout | deterministic vector plan composition | Endpoint matching was optimistic; fpeval wall 0.0116, opening 0.0039, validity 0.0156; geometry-aware matching is required |
| Geometry-aware vector scaffold | 256/64, 8 epochs | Honest proxy wall F1 0.3296 and opening F1 0.3250 at the selected checkpoint, but full-plan wall 0.0099, opening 0.0037, validity 0.0469; free queries remain ungrounded in room/wall hosts |
| Staged free-query room -> wall -> host-opening scaffold | 256/64, 8 epochs | Reject as an endpoint: composed wall 0.0030, opening 0.0219, r_val 0.0469; canonical targets align with fpeval, so free endpoint localization is the failure |
| Production-stage wall candidates + dedicated openings | zero-training cal64, evidence 0.2 | Pass on target-watertight eligible cases: wall 0.3683, opening 0.1211, room 0.2322, schema validity 1.0, watertightness 1.0 |
| Learned candidate selection + endpoint snapping | 256/64, 8-epoch trend screen | Active: matched against deterministic evidence threshold; next gate is conditional wall > 0.3683 without opening, room, schema, or watertight regression |
| TopoGED budget + multiscale filtration scorer | 256/64, 8-epoch matched screen | Active: global wall budget, candidate scores, junction-degree filtration descriptor, and Graphlet-style MSE/intersection/CDF objective against the deterministic candidate threshold |
| Descriptor-first candidate graph decoder | 64/32, 8-epoch fast mechanism screen | Active: stage-token descriptor prediction, candidate junction attention, descriptor conditioning, and separate exterior/interior budgets before any 256/64 confirmation |
| Score-only descriptor-first decoder | 64/32, 8 epochs | Best epoch 2 conditional Wall F1 0.3629 versus deterministic control 0.3539; later epochs collapsed, so score selection is positive but unstable. Refinement and budget top-K are harmful at this budget. |
| Anchored compact score residual | 64/32, 8 epochs plus smoke | Epoch 0 matches or slightly exceeds the evidence control, but the compact residual is still weak and later trend is not promotion-worthy; preserve as mechanism evidence and prioritize contract gaps |
| Many-query joint opening-host router | 64/32, 8 epochs with teacher-forced canonical queries | Historical dense-router control; does not beat detector fallback |
| Opening host hybrid MIta | 256 x 16 screen, 1,024 x 20 promotion, 5,126 x 16 confirmation | Active mechanism budget; requires dense/static-budget/shuffled-token controls |
| Predicted-channel static typed graph | 256 train, 8 epochs, 676k parameters, 512 updates | Diagnostic only; underbudgeted for claims about typed-channel mechanism quality |
| Predicted-channel evidence context + dynamic graph + dynamic stage heads | 512 train, 12 epochs, approximately 2.23M parameters, 1,536 updates | Superseded by the stage-MIta v2 candidate |
| Predicted-channel stage-MIta weak-stage graph | 512 train, 16 epochs, approximately 4.21M parameters, 2,048 updates | Source-router control |
| Predicted-channel source-router head | 512 train, 16 epochs, approximately 6.22M parameters, 2,048 updates | Active implementation budget with instance/program/furniture action stages |
| Room hybrid query-set MIta | 512 train, 16 epochs, approximately 16.33M parameters, 1,024 updates | Active mechanism screen; promotion uses 1,024 x 20 and five-seed confirmation uses 5,122 x 20 |
| Structure multi-scale MIta | 512 train, 12 epochs, approximately 2.90M parameters, 768 updates | Active mechanism screen; promotion uses 1,024 x 16 and confirmation uses 5,126 x 16 |
| Owner-boundary + expected-top-k soft count | 512 train, 12 epochs, calibration/holdout 64 | Active implementation budget; promotion uses 1,024 x 16 and full train uses 5,122 x 16 |

The direct-GT instance query improved the internal 256/64 screen from Room IoU
0.24685 to 0.25608 and instance F1 0.21706 to 0.24387 by epoch 8. A ten-epoch
run peaked at epoch 8 and regressed at epoch 9, so the promotion budget was
stopped on a documented plateau rather than filled to 24 epochs.

The full validation audit showed that direct query routing alone lowers plan
plausibility. A target-free explicit-plan validity/wall fallback removes that
regression. The resulting val491 route changes count MAE 3.59267->3.48473, Node
F1 0.71271->0.71562, Edge F1 0.46500->0.47079, and GED 11.24236->10.90020 while
Room IoU and Instance F1 remain statistically unchanged. A learned gate added no
significant benefit beyond this deterministic fallback, so the simpler mechanism
is retained.

The structure bridge was also tested at a larger budget. It plateaued four epochs
into the five-epoch continuation and changed calibration64 wall F1 from 0.45847
to 0.45654 and opening F1 from 0.11135 to 0.11037. This is evidence that the next
structure gain must come from representation or readout, not another epoch
increase.

The wall-centerline head is useful evidence: at threshold 0.8 with consensus
fusion it raises calibration64 wall precision from 0.357 to 0.702, but opening
F1 falls from 0.111 to 0.091 and instance F1 does not improve. The next budget is
therefore gated on a zero-training separate-layer room/wall/opening readout.

Layered readout kept the room path stable and confirmed the wall evidence, but
all existing opening sources failed to survive the final CAD opening metric.
The next budget is a trained `WallIntervalHead` with GT wall interval start,
length, presence, and type supervision; another threshold sweep is not justified.

Before executing any task, record the task card's pre-execution review:
matched control, expected effective
updates and pixels, the most relevant empirical curve, and the end-to-end
promotion budget. When local evidence cannot distinguish design choices,
refresh the 2025-2026 literature and official implementation before freezing
the run. A zero-training or frozen-checkpoint baseline must be measured before
training, and a negative one-to-three-epoch smoke run is never a scientific
rejection.

## Synthetic pretraining probe

The local fpgen pipeline generated 2,038 QA-passing synthetic plans
(1,630 train / 204 val / 204 test). The model reached 0.0995 synthetic-val room
IoU after eight epochs, but CubiCasa fine-tuning for six epochs peaked at only
0.0800 partition channel IoU, below the real-data baseline of 0.1132.

Direct synthetic pretraining is therefore not adopted. The result identifies a
domain-transfer issue in room vocabulary, rendering style, and room-area
distribution; the next attempt should use mixed real/synthetic batches or a
more CubiCasa-like dataset rather than simply generating more of the same.
