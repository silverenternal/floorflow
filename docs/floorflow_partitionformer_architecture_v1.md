# FloorFlow PartitionFormer Architecture v1

## Scope

This note records the architecture-first response to CE2EPlan. The local
CE2EPlan checkout is commit `774e670941eeda3e6677303ff79e8d517ae0d1c0`
(TVCG 2026). The purpose is to establish a stronger executable representation
before comparing downstream metrics. No RPLAN test row is used.

## CE2EPlan Source Audit

The released implementation has four important limits:

1. Every room is one six-value token
   `[exists, class, cx, cy, width, height]`.
2. The sequence has a fixed maximum of eight rooms.
3. The graph layer receives a complete room graph; topology is not predicted.
4. Sampling uses 1,000 diffusion steps, after which box alignment, room
   subtraction, and decoration resolve overlaps and uncovered regions.

Its strengths are a compact representation, a fully released pretrained
checkpoint, and deterministic post-processing. FloorFlow should preserve those
engineering advantages without inheriting the box bottleneck.

## Current FloorFlow Bottleneck

The deployed mask-partition path already avoids boxes and uses one NFE, but its
room decoder still emits an independent sigmoid mask for each query. The final
canonical readout resolves those masks by an argmax after training.

A 128-case validation audit of the seed-37 CAD64 checkpoint found:

| Diagnostic | Value |
| --- | ---: |
| Footprint pixels under more than one 0.5-threshold mask | 0.1830 |
| Cases with any threshold overlap | 1.0000 |
| Footprint pixels without any active room mask | 0.0451 |
| Cases with more than 5% uncovered footprint | 0.2969 |
| Mean active masks per footprint pixel | 1.1432 |

This is an architectural mismatch. Rooms form one interior partition, while
the decoder is trained as several independent binary segmentation problems and
their conflicts are repaired later.

Evidence:
`outputs/reports/rplan_partition_diagnostic_cad64_control4096_seed37_v1.json`.

## PartitionFormer Contract

The new opt-in decoder changes the output contract:

```text
boundary memory
  -> variable room queries
  -> query competition at every room pixel
  -> one owner per pixel
  -> selected-query owner map
  -> canonical polygons and shared room boundaries
```

The decoder returns:

- ordinary per-query sigmoid masks for compatibility;
- `room_instance_partition_logits` as a joint query distribution;
- `room_instance_partition_probabilities` summing to one across room queries.

The second decoder layer suppresses attention to pixels owned by another room;
unclaimed and exterior pixels remain visible so a query can still grow. This
uses the current prediction rather than a fixed complete graph or an oracle
room adjacency.

Training matches room queries to target instances with the existing Hungarian
assignment. The matched query index, not only the room class, supervises every
target-room pixel through a partition cross-entropy term. The canonical reader
consumes the joint owner and does not recover room identity through an
independent sigmoid threshold.

Old checkpoints remain valid because the new branch and parameters are absent
when `enable_partition_consistent_queries=false`.

## Architectural Difference

| Capability | CE2EPlan | FloorFlow PartitionFormer |
| --- | --- | --- |
| Room cardinality | fixed eight | variable query count within 12 |
| Room geometry | axis-aligned box | mutually exclusive mask plus canonical polygon |
| Pixel ownership | resolved after box sampling | joint query owner during decoding |
| Room graph | complete fixed graph | predicted mask/boundary evidence |
| Uncovered interior | repaired by box subtraction | represented by the joint partition |
| Sampling | 1,000 diffusion steps | one forward NFE |
| CAD output | decorated boxes | audited room/wall/opening/furniture layers |

The table is a capability contract, not a performance claim. A performance
claim requires the matched screen and multi-seed evidence below.

## Implemented Code

- `src/floorflow/heads/room_mask_decoder.py`
- `src/floorflow/heads/room_mask_losses.py`
- `src/floorflow/model/unified_floorplan_generator.py`
- `src/floorflow/inference/mask_partition_readout.py`
- `src/floorflow/evaluation/rplan_mask_partition.py`
- `configs/rplan_generation_floorflow_partitionformer_v1.json`
- `scripts/diagnose_rplan_room_partition_v1.py`

## Contract Smoke

The real-data smoke in
`outputs/train/floorflow_partitionformer_contract_smoke_seed2905_v2`
passed:

- finite forward, loss, backward, and optimizer step;
- positive partition supervision (`9,126` matched room pixels in the train
  batch and `820` in validation);
- one canonical plan assembled;
- no vector validation errors; and
- exact token round trip.

The smoke is an integration result only and cannot be used as a quality claim.

## Matched Screen

The first scientific screen is:

- 256 train / 64 validation cases, disjoint validation offset 256;
- 8 epochs, batch 8, seed 2905, one NFE;
- identical ResNet18 backbone, CAD64 query banks, augmentations, optimizer, and
  readout;
- control: independent room masks;
- candidate: joint query partition plus one matched partition-loss term.

Promotion requirements:

- Room IoU no more than 0.01 below the control;
- instance F1 and partition coverage improve together;
- threshold overlap falls below 0.02 before canonical repair;
- validity remains 1.0 and conditional watertightness remains at least 0.95;
- token round trip remains exact.

If the probe fails, profile assignment stability, partition entropy, and the
effect of masked attention before changing the loss weight. A loss sweep alone
cannot establish the architecture.

### Two-Seed Result

Seeds `2905` and `37` completed the matched `256/64`, eight-epoch screen.
Canonical Room IoU improves by `+0.02892`, instance F1 by `+0.01635`, type
accuracy by `+0.03359`, and count MAE by `-0.06250` in the two-seed mean.
Room IoU, F1, and type accuracy improve on both seeds; count improves on one.

Structural changes are consistent with the representation:

| Metric | Control mean | PartitionFormer mean | Delta |
| --- | ---: | ---: | ---: |
| Node F1 | 0.87553 | 0.89750 | +0.02197 |
| Edge overlap | 0.38774 | 0.51085 | +0.12311 |
| GED | 4.73438 | 3.79688 | -0.93750 |
| Validity | 1.00000 | 1.00000 | 0.00000 |
| Conditional watertightness | 0.96875 | 1.00000 | +0.03125 |
| Dangling-endpoint rate | 0.02295 | 0.01473 | -0.00822 |

The first structural run exposed a separate vectorization defect: a
GeometryCollection produced by clipping the largest living-room query was
discarded because the reader handled only Polygon and MultiPolygon. Shrinking
the query to its largest polygonal member restored the shared partition. A
second issue came from secondary clipped fragments; joint-partition readout now
reassigns dropped fragments through nearest primary-owner completion. After
these fixes, both seeds reach watertightness `1.0`.

Evidence:
`outputs/reports/floorflow_partitionformer_twoseed_v1.json`.

The mechanism passes the small architecture gate and advances to the
predeclared 1024/256, 12-epoch, two-seed promotion screen.

## Promotion Screen

The initial hard owner attention did not scale. On seed 2905 it reduced
canonical Room IoU from `0.59147` to `0.56320`. Keeping joint owner supervision
and readout while disabling exclusive cross-attention recovered seed-2905 Room
IoU to `0.60607`. A no-owner-CE ablation confirmed that owner supervision helps
object identity but must not be paired with an early hard attention constraint.

The promoted candidate therefore uses:

- joint owner probabilities and jointly exclusive canonical readout;
- Hungarian-matched owner CE;
- ordinary multi-scale query attention, without hard room-core suppression.

At 1024/256, batch 8, 12 epochs, the two matched seeds give:

| Metric | Control mean | PartitionFormer mean | Delta |
| --- | ---: | ---: | ---: |
| Canonical Room IoU | 0.60008 | 0.60385 | +0.00376 |
| Canonical instance F1 | 0.38715 | 0.40435 | +0.01720 |
| Canonical type accuracy | 0.70725 | 0.69840 | -0.00885 |
| Canonical count MAE | 0.97461 | 0.87305 | -0.10156 |
| Structural Room IoU | 0.61586 | 0.61452 | -0.00133 |
| Structural instance F1 | 0.39308 | 0.41631 | +0.02323 |
| Node F1 | 0.89053 | 0.91883 | +0.02830 |
| Edge overlap | 0.57298 | 0.61751 | +0.04453 |
| GED | 3.55664 | 3.02930 | -0.52734 |
| Conditional watertightness | 0.99219 | 0.99805 | +0.00586 |
| Dangling-endpoint rate | 0.01845 | 0.01229 | -0.00616 |

Validity remains `1.0`. Canonical F1 improves on both seeds; canonical Room IoU
improves on one and remains within the predeclared `0.01` tolerance on the
other. The owner partition has zero pre-readout overlap and gap in both seeds.
Type accuracy retains a small `-0.00885` mean regression and is reported rather
than hidden.

Evidence:
`outputs/reports/floorflow_partitionformer_promotion_twoseed_v1.json`.

The gradient audit shows that adding owner CE raises the identity-decoder and
spatial-encoder gradient norms by only `7.85%` and `14.88%` on a real batch.
The partition term is therefore integrated rather than dominating the existing
objectives.

The no-hard-attention candidate advances to the 4096/512, batch-16, eight-epoch
scale screen.

## Scale Correction

The dense owner-CE candidate did not survive scale. At 4096/512 it reached
canonical Room IoU `0.61927` against the current-code control `0.66979`.
Training-set owner accuracy was high while validation geometry regressed,
which is assignment memorization rather than missing capacity.

Three subsequent controls isolate the useful mechanism:

1. Hard attention is rejected; it caused a large 1024 Room IoU drop.
2. Dense owner CE is rejected for the main model at 4096.
3. A confidence-gated joint owner readout is retained: independent query
   confidence defines legal room support, while joint query competition resolves
   the unique owner inside that support.

The current-code 4096/512 two-seed comparison is:

| Metric | Control mean | Confidence-gated readout | Delta |
| --- | ---: | ---: | ---: |
| Canonical Room IoU | 0.66679 | 0.66780 | +0.00101 |
| Canonical instance F1 | 0.50460 | 0.50570 | +0.00110 |
| Canonical type accuracy | 0.75157 | 0.76092 | +0.00935 |
| Canonical count MAE | 0.85547 | 0.87207 | +0.01660 |

The quality change is small but stable in F1. The topology gain is not yet
robust across both seeds, so the architecture is not claimed to dominate the
control. It is a cleaner executable contract and remains a candidate mechanism.

The following mechanisms were explicitly rejected by matched screens:

- hard owner attention;
- dense owner cross-entropy at 4096;
- conflict-only owner cross-entropy with label smoothing at 1024;
- graph-refined class logits at 1024, which degraded type accuracy and graph
  metrics despite connecting the existing graph refiner to the identity head.

Room adjacency is now derived directly from the owner raster before
vectorization, so the canonical CAD graph does not have to recover adjacency
from separately quantized polygon contacts. This is an integration contract;
its structural effect requires a promoted checkpoint rather than a report-only
interpretation.

The next architecture stage is a boundary-anchored variable room-program graph
that changes the room set before mask decoding, rather than adding another
post-decoder head.

## Boundary-Edge Graph Promotion

`BoundaryEdgeGraphEncoder` is now used as the room-decoder memory for the
promoted architecture arm. It represents boundary segments directly with
length, direction, turn, arc position, and entrance markers. This is the
architectural counterpart to CE2EPlan's flattened 80-coordinate boundary
condition.

At 1024/256, 12 epochs, batch 8, seeds `2905` and `37`, with the same
confidence-gated owner readout and no owner CE or hard attention:

| Metric | Current control | Boundary graph | Delta |
| --- | ---: | ---: | ---: |
| Canonical Room IoU | 0.63106 | 0.63183 | +0.00077 |
| Canonical instance F1 | 0.46286 | 0.48494 | +0.02208 |
| Canonical type accuracy | 0.70885 | 0.75280 | +0.04395 |
| Canonical count MAE | 0.91992 | 0.89453 | -0.02539 |
| Structural Room IoU | 0.64486 | 0.63968 | -0.00518 |
| Structural instance F1 | 0.47091 | 0.49220 | +0.02129 |
| Node F1 | 0.90521 | 0.92720 | +0.02199 |
| Edge overlap | 0.56345 | 0.60166 | +0.03821 |
| GED | 3.42773 | 3.12109 | -0.30664 |
| Conditional watertightness | 0.18945 | 0.19141 | +0.00195 |

Instance F1, type accuracy, structural F1, Node F1, Edge overlap, and GED
improve on both seeds. Canonical Room IoU is seed-balanced and the structural
Room IoU change is within the `0.01` tolerance. The ownership/dangling rates are
reported separately because the quality-first threshold readout does not solve
watertight closure by itself.

Evidence:
`outputs/reports/floorflow_boundarygraph_twoseed_v1.json`.

This is the first architecture arm in the current line that improves six
quality/topology signals toward the same direction on two matched seeds without
using oracle controls. It advances to 4096/512; the scale result remains the
deciding gate for a main claim.

### 4096 Scale Result

The 4096/512, batch-16, eight-epoch two-seed promotion preserves the quality:

| Metric | Control mean | Boundary graph mean | Delta |
| --- | ---: | ---: | ---: |
| Canonical Room IoU | 0.66679 | 0.66556 | -0.00124 |
| Canonical instance F1 | 0.50460 | 0.51480 | +0.01020 |
| Canonical type accuracy | 0.75157 | 0.79450 | +0.04294 |
| Canonical count MAE | 0.85547 | 0.84668 | -0.00879 |
| Structural Room IoU | 0.67674 | 0.67243 | -0.00431 |
| Structural instance F1 | 0.51224 | 0.52101 | +0.00877 |
| Node F1 | 0.90540 | 0.92948 | +0.02408 |
| Edge overlap | 0.57150 | 0.60624 | +0.03474 |
| GED | 3.43066 | 3.11523 | -0.31543 |
| Conditional watertightness | 0.20313 | 0.26172 | +0.05859 |
| Dangling-endpoint rate | 0.02008 | 0.02129 | +0.00121 |

F1, type, instance F1, Node F1, Edge overlap, and GED improve on both seeds.
Room IoU remains within the predeclared `0.01` tolerance, and validity remains
`1.0`. Dangling endpoints are slightly worse and are retained as a limitation.

This passes the architecture promotion gate and defines the next test:
five-seed 4096 stability followed by full validation.

### Five-Seed Readout Closure

The five-seed scale screen exposed two independent issues that were not model
capacity failures:

1. Independent room-polygon rasterization could turn a sub-pixel,
   exterior-connected vector channel into an apparent interior hole. The raster
   diagnostic now fills this seam only when the vector room union is already
   hole-free; a true vector hole still fails.
2. Some rooms had two large support components separated by only 2.0-3.61
   pixels. `bridge_fill` connects substantial components with one-pixel shortest
   bridges, while the adaptive policy tries threshold, hole fill, bounded
   bridging, and full partition fill in that order.

The adaptive safety-first readout is used for both quality and structure in the
five-seed table:

| Metric | Control mean | Boundary graph mean | Delta |
| --- | ---: | ---: | ---: |
| Room IoU | 0.66258 | 0.66807 | +0.00549 |
| Instance F1 | 0.49828 | 0.51950 | +0.02122 |
| Type accuracy | 0.75305 | 0.81080 | +0.05775 |
| Count MAE | 0.87148 | 0.85469 | -0.01680 |
| Structural F1 | 0.50493 | 0.52793 | +0.02299 |
| Node F1 | 0.89485 | 0.92449 | +0.02964 |
| Edge overlap | 0.57660 | 0.62203 | +0.04543 |
| GED | 3.45625 | 3.03242 | -0.42383 |
| Conditional watertightness | 0.93906 | 0.99961 | +0.06055 |

Validity remains `1.0` for both arms. Instance F1, type accuracy, Node F1, edge
overlap, GED, and conditional watertightness improve on all five seeds. Room
IoU improves on three seeds; its paired 95% t-interval
`[-0.00995, 0.01567]` crosses zero, so no significant mean Room IoU gain is
claimed. With five pairs, the exact two-sided Wilcoxon p-value for an all-5/5
direction is `0.0625`; the rank-biserial effect is `1.0` for the consistent
metrics. Five seeds alone do not justify a `p < 0.05` statement.

Dangling endpoint rate remains a limitation: control `0.02057`, candidate
`0.02168`. The candidate therefore passes the graph/watertight closure but not
the next architecture target of at most `0.02`.

Evidence:
`outputs/reports/floorflow_boundarygraph_scale_fiveseed_adaptive_v1.json`.

## Next Architecture Stages

1. Derive shared room boundaries, wall junctions, and opening hosts from the
   same owner map so topology is not reconstructed from unrelated heads.
2. Add a finite ordered CAD action transformer for envelope, program, rooms,
   walls, junctions, openings, furniture, and validation.
3. Match CE2EPlan on parameters, training exposures, and NFE in a separate
   efficiency table. Do not compare only the pretrained checkpoint to an
   underbudgeted FloorFlow run.

## External Research Anchors

- CE2EPlan, TVCG 2026, local source commit
  `774e670941eeda3e6677303ff79e8d517ae0d1c0`.
- TLC-Plan, arXiv `2602.07100`, Feb 2026: two-level vector hierarchy and
  topologically valid autoregressive generation.
- FMLM, arXiv `2604.04859`, CVPR 2026: markup representation for conditional
  floorplan generation.
- Raster2Seq, arXiv `2602.09016`, SIGGRAPH 2026: labeled polygon sequences for
  reconstruction.
- CAGE, arXiv `2509.15459`: edge-continuity and dual-query topology decoding.
- PolarSym, arXiv `2608.11793`, Aug 2026: direction/distance geometric attention
  for CAD parsing.
- Flatten the Complex, arXiv `2601.17733`: compositional k-cell particles share
  interface latents across vertices, edges, and faces.
- FloorplanVLM, arXiv `2602.06507`: structured output preserves global topology
  and supports non-Manhattan geometry.
- SALI-FP, arXiv `2609.25615`: evidence-gated local revisions and reviewable
  vector/relation records.
- Topology-Preserving Polygon Augmentation, arXiv `2603.14764`: index-space
  adjacency repair for polygons with interior voids.

These works motivate complementary representations but none replaces the
PartitionFormer contract. No external leaderboard claim is made in this note.
