# Chip Architecture Correctness Audit v244

Date: 2026-11-02

This pass covers the chip floorplanning path only. No floorplan/room model was
used in the conclusions. No training or quality metric is claimed.

## Confirmed defects and repairs

### Route-field coordinate compression

`SpatialOccupancyCoupler` applied `sigmoid` to block centers even though flow
centers are already normalized coordinates. This compressed a full layout into
roughly `[0.269, 0.731]` before rasterization and `grid_sample`.

Repair: use the normalized centers directly, map only the final query to
`grid_sample`'s `[-1, 1]` domain, and normalize block centers and extents with
the same target-free pin chart used by p2b route fields.

### Relation-specific cross-membership leak

`TypedCrossAttention.forward_relations` first attended over the complete
structural memory using the unrestricted base mask, then added relation-masked
branches. A non-member cluster could therefore influence a block before the
relation mask was applied.

Repair: relation-specific fusion now starts from a zero tensor and composes only
the four explicitly masked families.

### Dynamic edge scale mismatch

`StateEdgeRelay` compared normalized block states with raw physical pin
coordinates and absolute area, so a uniform physical scale change altered its
residual.

Repair: the shared sparse edge cache exposes normalized area fractions and
pins normalized by the same flow scale. Raw coordinates remain available only
for spatial rasterization.

### Structural grouping scale mismatch

`mean_gap` and the adjacency threshold in `structural_relation_features` used
physical units while the other six group moments were chart-normalized.

Repair: gap and adjacency thresholds are normalized by the same chart scale.
The complete seven-feature group moment is now scale-equivariant.

### Shape-mode dead end

`ShapeModeHead` generated multiple rho candidates, but sampling exported them
only as diagnostics. The final legalized layout never consumed them.

Repair: the head now predicts per-mode selection logits, receives WTA winner
cross-entropy supervision, and can apply its selected mode to the sampled block
state. The zero-initialized mode head still selects the identity mode at
initialization.

The MIB-sharing pass also no longer overwrites anchored/fixed rho values while
propagating a shared ratio to free members.

### One-way hierarchy state

L0/L1/L3 heads did not observe the evolving L2 block layout. An optional
`ObservedHierarchyCoupler` now aggregates current block geometry into observed
L0 bbox, L1 moments, and port-local block centers. Three final projections are
zero-initialized, so the old model is exactly recovered until training opens
the path.

### Port-to-block geometric disconnect

The L3 port head predicted `block_center - pin` offsets, but the existing
`PortConsensusCoupler` did not know fixed pin coordinates and therefore could
not enforce `block_center = pin + offset`.

Repair: hierarchy context now carries the normalized pin chart and valid-pin
mask. Port consensus adds the implied geometry residual behind a separate
zero-initialized scale. The original neural residual remains independent.

### Double-normalized endpoint displacement

Endpoint compatibility subtracted two already-normalized flow latents and then
divided the result by `sqrt(total_area)` a second time. This systematically
underweighted legalizer displacement on large designs.

Repair: displacement and legalizer-target errors now stay in the normalized
latent chart and are divided by the design scale exactly once when response
labels are generated.

### Routing proxy lacked obstruction/detour response

The previous routing proxy splatted b2b/p2b demand at route midpoints but did
not see whether a route crossed macro occupancy. This left p2b/pin-demand gains
structurally unable to answer the routed-WL negative-transfer concern.

Repair: `batch_routing_proxy_terms` now samples both L-shaped routes per edge
through the differentiable occupancy field and adds a normalized
`route_obstruction` term. It uses the existing edge cache, occupancy raster and
normalization conventions, and remains invariant to translation and uniform
physical scaling.

### Incorrect local pin spread

Raw local geometry subtracted an already normalized pin centroid from physical
pin coordinates and then divided again. The resulting spread was neither local
nor scale-equivariant.

Repair: pin coordinates and their weighted centroid are normalized together
before the weighted RMS spread is computed.

### Corrected mechanisms left disabled

The corrected group-moment, relation-specific, dynamic graph-input and spatial
route-field branches are now explicit in `configs/chip_architecture_mainline_v4.json`.
They are no longer hidden repository code that every checkpoint silently leaves
off. Their previous negatives remain retained, but are labeled
implementation-contaminated rather than intrinsic mechanism failures.

### Invalid internal hyperedge configuration

Internal FloorSet caches do not contain `net_features`, `net_block_indices`,
or net incidence. The mainline configuration nevertheless enabled the
hyperedge branch, which fails closed when executed.

Repair: hyperedge conditioning is external-graph-only in the branch registry.
The internal mainline disables it explicitly.

## Consequence for retained negatives

The route-aware M55/M61A failures and structural-relation group-moment failures
were obtained with at least one of the coordinate, scale, or mask defects above.
They cannot be used as clean evidence that those mechanisms are intrinsically
unhelpful. Their status changes to `implementation-contaminated`, not `closed`.

Shape modes have never entered a legalized candidate pool. Their existing screen
therefore evaluated the auxiliary loss only; it was not a mechanism result on
the final endpoint.

## Weak-metric response contract

| Weak metric / reviewer concern | Structural answer | Falsification condition |
|---|---|---|
| b2b HPWL gap | Remove relation base-memory leakage, restore scale-equivariant dynamic edge inputs, and condition b2b on current L2 geometry. | Corrected b2b branch still improves only raw proxies and worsens legalized b2b/cost tail. |
| area / rho gap | Shape modes now select and mutate the sampled endpoint; MIB sharing preserves anchored shapes; endpoint displacement is normalized once. | Selected shape mode does not improve legalized area/rho or expands the candidate-pool oracle. |
| p2b→routed disconnect | Unify block/pin/route charts, add the pin+offset geometry residual, and apply a position trust radius. | Corrected route/port path still has non-positive routed-WL rank agreement on fresh designs. |
| grouping offset | Relation-specific attention is no longer contaminated by unrestricted structural memory; structural gap is chart-normalized; observed L0/L1 states consume L2 geometry. | Corrected grouping branch shows no legalized grouping gain or oracle expansion. |
| hierarchy drift | L0/L1/L3 heads receive current block-derived observations through zero-initialized adapters. | Corrected coupling cannot be distinguished from its off control after matched training. |
| opening / MIB / highest-complexity claims | No new claim is made. MIB remains diagnostic where the metric has no discrimination; opening is outside this repair. | Any claim requires a separately identified positive atom. |

## Verification

Focused contracts cover:

- no sigmoid compression of normalized centers;
- block and pin geometry share one chart;
- relation branches have zero cross-membership sensitivity;
- dynamic edge relay and structural moments are scale-equivariant;
- shape-mode selection is trainable, deterministic, and applied at sampling;
- observed hierarchy coupling is zero-gated, differentiable, and checkpointed;
- internal configurations reject external hyperedge conditioning.

## Shape-mode consumer probe

A CPU legalized probe used the existing S7 checkpoint on four dev cases
(21/30/40/50 blocks), NFE4, seed 19 and one shared legalizer policy.

| Arm | Cost | V_rel | Area gap | Rho MAE | HPWL gap |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.11514 | 0.02632 | 0.07243 | 0.25639 | 0.03282 |
| Mode mean | 1.10609 | 0.02216 | 0.07526 | 0.25501 | 0.03380 |
| Per-case oracle | 1.09764 | 0.01974 | 0.07132 | 0.25540 | 0.03242 |

All 20 baseline/mode endpoints are legalized-feasible. Mode mean improves cost
and V_rel but regresses area and HPWL, so the branch is retained only as an
atom-specific candidate generator. The oracle improvement shows that the
candidate pool now expands; it is not a deployable mean or SOTA claim.

Report: `outputs/reports/chip_shape_mode_consumer_probe_v1.json`.

### Selector-only legalized teacher

The zero-initialized mode selector was trained on 12 train cases using only
legalized hard cost as the winner label. Only `mode_score` and `mode_bias` were
trainable: 677 parameters. On the four disjoint dev cases, the selected mode
changed the final legalized endpoint versus baseline by:

- cost `-1.74%`;
- V_rel `-25.0%`;
- area gap `-3.86%`;
- rho MAE `-1.63%`;
- HPWL gap `-6.81%`.

All legalized endpoints remained feasible. This is a retained CPU candidate, not
a promotion: the confirmation set must be expanded and case-disjoint. The
selector’s train-split teacher accuracy was only about 50%, so generalization
is still unproven.

Reports:

- `outputs/reports/chip_shape_mode_selector_train_v1.json`
- `outputs/reports/chip_shape_mode_selector_probe_v1.json`

### Selector collapse and repair

Both the 12-case and 80-case selector checkpoints selected mode 0 on all 92
development cases. The 80-case selector regressed the main legalized metrics,
while the all-mode oracle remained positive on every reported metric. This
rejects selector promotion but confirms the candidate generator.

Root cause: the selector only saw a hidden summary that did not expose the
target-free differences among candidates. Its zero-initialized final score layer
also made the first descriptor gradient unreachable.

Repair:

- add per-mode b2b, p2b, overlap, area, HPWL, MIB, boundary, and rho-delta
  descriptors;
- encode those descriptors before mode scoring;
- add a small-random score head with a mode-0 identity prior so gradients are
  reachable while initialization still selects the conservative mode;
- keep descriptors out of the rho-residual generator, so they only control
  selection.

The repaired selector still requires matched training and a disjoint
confirmation before promotion.

Shape modes now also have independent spatial support budgets
(`10% / 25% / 50% / 100%` blocks). This turns modes into increasing-risk shape
hypotheses rather than identical global rho offsets with different magnitude.

Both learned and Pareto selectors now include the unchanged baseline as
candidate zero. The learned selector has an independent baseline feature score
and a conservative positive prior; it is no longer forced to choose a shape
mode on every case.

The ordinary shape-mode training loss now uses the same `baseline + modes`
candidate set as inference. Winner-takes-all and selection cross-entropy can
therefore choose the unchanged endpoint, eliminating the former train/inference
candidate-space mismatch.

A second implementation defect was found during the Pareto probe: selecting
the baseline candidate still triggered an extra hard projection. Baseline
selection now skips shape projection per case, so a safe gate can preserve the
parent endpoint exactly.

Dev92 negative reports:

- `outputs/reports/chip_shape_mode_selector_dev92_v1.json`
- `outputs/reports/chip_shape_mode_selector80_dev92_v1.json`

The failed selector-only checkpoints were removed after their negative results
were recorded.

### Pareto-safe fallback

The candidate set now includes the baseline at index zero. A target-free gate
accepts a mode only when at least one structural descriptor improves by at
least 1% and no descriptor regresses by more than 1%.

- The initial gate selected 10/32 cases and regressed legalized cost because
  tiny proxy improvements did not transfer.
- With the material-improvement margin, 31/32 cases kept the baseline and one
  selected a mode. Cost changed by `+0.0004%` and V_rel was unchanged.

The gate is retained as a safe fallback, not as a positive mechanism. Shape-mode
oracle gains remain the reason to keep the candidate generator.

Reports:

- `outputs/reports/chip_shape_mode_pareto_support_dev32_v1.json`
- `outputs/reports/chip_shape_mode_pareto_support_dev32_v2.json`
