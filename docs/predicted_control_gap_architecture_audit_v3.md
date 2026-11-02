# Predicted-Control Gap-to-Architecture Audit v3

Date: 2026-11-02

## Scope

This audit starts from the authoritative full491 predicted-control gaps and
traces each directly addressable weakness into the promoted network. It does not
promote a metric. The pre-repair baseline remains Room IoU `0.327585`, Instance
F1 `0.281759`, Node F1 `0.727429`, Edge overlap `0.361825`, and GED `7.682281`
from `outputs/reports/predicted_control_full_stage_val491_gedfixed_v2.json`.

## Findings and Repairs

### Room IoU and Instance F1

`preserve_room_stage=True` previously returned the raw control prior and discarded
the typed-channel room logits. The entire learned all-channel room path therefore
had no path to the final room partition. Stage preservation now suppresses only
the room-stage direct replacement while retaining the typed-channel head output.

The room-stage candidate is separately blended into the query decoder's
prototype evidence with the existing bounded `residual_scale`. This gives
`stage_rooms` a live gradient path without allowing it to replace the partition.

The all-channel head also computed a background query but replaced its spatial
logits with one global scalar. The background query now contributes a small
spatial residual to the final logits, so exterior and room-boundary errors can
supervise the previously dead background channel.

The query-set count head initialized all `0...32` room counts uniformly, implying
16 selected queries before seeing an image even though the audited mass is
concentrated at 5-9 rooms. A weak center-8 initialization now prevents the owner
map from starting in a diffuse state.

Class-count supervision previously updated only the count head; it did not force
the actual `sum(query presence * class probability)` to match that head. A
differentiable cardinality-consistency term now connects the two consumers. Its
weight is independently configurable for ablation.

Room graph message passing also treated every query as a live neighbor. The
conditioner now weights pair messages by endpoint presence before row
normalization, preventing inactive or duplicate query slots from diluting active
owner, class, and mask residuals.

The room decoder's existing `fusion_scale` had no consumer in the promoted
predicted-control path. Mask probability now pools dense prototype class logits
inside each query and adds that region evidence to the final class logits through
the existing zero-initialized scale. This gives Type accuracy and Instance F1 a
direct mask-to-class path without new parameters.

Metric ownership:

- Room IoU: typed-room residual, background query, stage-room context, and
  count-aware initial routing.
- Instance F1: the same room prototype and query decoder path.
- Count/type: count prior plus query-to-class-cardinality consistency.

### Node F1, Edge overlap, and GED

The dense canonical type-edge objective used unnormalized overlap sums followed
by `1 - exp(-20x)`. Normal room support therefore saturated nearly every pair to
one, and the final clamp removed gradient at negative edges. The objective now
normalizes contact by the geometric mean room area and applies BCE directly to
the normalized contact probability.

The canonical 2 px tolerance mapping is retained: two blank raster cells remain
adjacent, while larger separations do not. A three-room attack verifies that an
otherwise-correct partition with an extra 0-to-2 contact receives a higher loss.

Metric ownership:

- Edge overlap: normalized positive/negative contact supervision.
- GED: edge insertion/deletion pressure from the same canonical type graph.
- Node F1: unchanged fuzzy-OR node projection.

### Silent Consumer Disconnects

An instance room decoder or action-token refresh configured without the ordered
CAD action graph previously built successfully and then silently did nothing.
Both configurations now fail closed at model construction.

Room-query attention also converted the previous masks into hard booleans. Early
mask mistakes therefore became permanently invisible. The query decoder now uses
finite additive penalties for spatial and partition-owner attention; confident
competition is suppressed while the floor remains recoverable.

The all-channel head exposed per-pixel reliability, but the promoted residual
fusion ignored it. Residual logits are now weighted by `0.75 + 0.25 ×
reliability`: reliable regions retain the learned residual scale, while
uncertain regions are reduced by at most 25%. The gentle floor avoids
over-suppressing old checkpoints under the high-class-count entropy regime.

Validation route diagnostics now collect reliability, mask-region class
evidence, room-stage candidate delta, and room-adjacency probability alongside
route health. The collector queues device scalars and performs one stacked host
transfer instead of synchronizing once per metric.

### Opening Host Coverage

`CadStageRuntime` retained only the product of opening foreground and family
probability. Opening-host query extraction consumed that product while canonical
evaluation consumed independent foreground, so family uncertainty suppressed
otherwise valid queries before routing. Runtime now retains foreground, family
logits, and their product under separate control keys. Query extraction uses the
independent maps; compatibility consumers retain the product.

The frozen-feature cache now serializes all three maps and the geometry junction.
Its schema contract advances so a cached run cannot silently restore the old
product-only control semantics.

Routed opening hosts are now projected through a shared inference consumer that
maps global candidate-wall indices to the selected final-wall order. Active
openings targeting unselected walls fail closed, and a plan with no selected
walls returns an empty opening layer instead of indexing an empty mapping.

### Wall and Junction

Runtime already computed a geometry junction map but exported it only through an
optional historical flag. Typed action graphs therefore fell back to wall
probability in the default configuration. Geometry junction is now always bound;
the flag no longer discards already-computed evidence.

Geometry and structure-head junction probabilities are fused before export, so
both independently supervised topology sources feed the same typed consumer.
Composed candidate plans now record incident walls in final local wall order
instead of leaking global candidate indices.

Wall-candidate caches advance from v7 to v8 because the same layer checkpoint now
produces different action-token inputs. Old caches are intentionally rejected.

### Furniture

The furniture query head accepted a class-density map in its API, but the
promoted furniture stage never supplied it. Dense furniture controls now drive
both class-concentration budget adjustment and mask-region class evidence.
Prototype evidence is pooled only after masks are available; it is not appended
to query attention memory, preserving the existing decoder ordering.

The furniture fusion uses the existing zero-initialized room-decoder fusion
scale, while missing dense controls remain a supported legacy call.

### Predicted-Control Editing

The predicted edit-state adapter previously applied the room count gate only to
`room_exists`. Rejected query slots retained their identity tokens, foreground
class logits, and adjacency rows, leaving ghost objects available to graph and
transition consumers.

All query-indexed state is now masked from one shared active-slot mask: tokens
are zeroed, existence and adjacency are saturated negative, foreground classes
are saturated negative, and the no-object class is set for inactive slots.
Adjacency shape and class-slot alignment fail closed before masking.

Strict vector projection previously classified an opening as edited only when its
center pixel landed inside the edit region. An interval whose center was outside
but whose width crossed the region could therefore remain as a stale outside
object. Opening membership now samples the full interval along its nearest host
wall, using before and after wall plans separately.

### RPLAN Target Limits

Room tensors and room count now draw from the same deterministic largest-room
subset. Previously `room_count` used the full plan while `room_exists` and
adjacency were capped, which could emit an out-of-range count target. Adjacency
indices are also built only from retained rooms, preventing writes past the
tensor limit.

Token targets use the same limiter, so dense query and token branches see one
room set. The source room count remains in the target batch for truncation
auditing and verifier hashing.

Local CE2EPlan/RPLAN audit covers 56,053 train and 12,018 validation rows. Room
counts range from 4 to 8 in both splits, so the latent `max_rooms=12` contract bug
does not reduce current data coverage.

Quality arbitration could previously return zero selected rooms even though the
generation decoder requires at least one. A positive query budget now retains at
least the highest-scoring candidate when all quality probabilities fall below the
minimum. Existence, quality, and budget shapes fail closed.

Entrance geometry used the smallest covariance eigenvector, which is the short
axis of `entrance_expand`, while naming it the tangent. It also prioritized
nearest boundary over parallel alignment. The long eigenvector is now used and
parallel alignment is the primary host criterion. On the first 5,000 local
validation entrances, this changes the selected boundary in 3,919 cases
(78.38%).

The RPLAN target-contract auditor checks room count/prefix, adjacency symmetry
and inactive slots, room point validity, wall/opening host identity, junction
degree, and boundary validity. Full train passes `56,053/56,053` and validation
passes `12,018/12,018` with zero violations. Room counts are 4-8 in both splits.
All 18 RPLAN configs also pass the static model-target capacity contract.

Token-target audit passes all `56,053` train and `12,018` validation cases at
maximum length `512`. Full train maximum length is `326`; the original
4,096-case vocabulary has `331/10,841,401` OOV (`3.05e-5`). A train-only closure
preserves all base token IDs and appends 67 unseen structural tokens, producing a
412-id model vocabulary. On full validation, OOV falls to `12/2,325,128`
(`5.16e-6`); the remaining unseen combinations are val-only and are not inserted
to avoid leakage.

Two CPU contract smokes cover the real SOTA configurations. The dense SOTA config
completes model construction, one train case, one validation case, backward, and
report generation. Its untrained random output is rejected by the canonical plan
validator as expected. The token stack with the 412-id closed vocabulary also
completes forward/backward with zero OOV and zero train grammar violations; its
teacher/vector contracts round-trip exactly. Smoke checkpoints were deleted after
retaining only the reports.

Canonical token roundtrip is audited on all `12,018` validation plans under the
1/16 fixed-point coordinate contract. Every plan preserves structural counts and
re-encodes to the same token text. Maximum quantization error is `0.022727 <
0.03125` half-step and maximum token length is `324`. Exact floating fingerprint
match is intentionally not the gate because fixed-point geometry is lossy.

## Verification

- `1199 passed, 5 skipped` for the complete non-chip, non-NLP suite including
  `test_package_layout.py`.
- Focused room, graph, predicted-control training, and live-path tests pass.
- Ruff passes for every touched Python file.
- All touched Python files remain below 600 lines.
- The promoted model remains `6805899` parameters; no checkpoint keys changed.
- The matched v3 dense-bridge control checkpoint loads with `strict=True` and
  zero missing keys.

## Dynamic Room Consumer Candidate v4

The metric split directly identifies one missing consumer: Node F1 `0.727429`
is much higher than Edge overlap `0.361825`, and 75 cases have Node F1 above
`0.8` while Edge overlap remains below `0.5`. The previous adjacency head saw
only query embeddings, so it could not explicitly represent whether the final
room masks actually touch. Canonical Edge and GED, however, are computed from
that room geometry.

The v4 candidate adds two independently switchable mechanisms:

- `room_instance_adaptive_query=True` exposes the existing image-conditioned
  query-reference sampler inside the predicted-control room decoder. Its
  reference head previously received no first-step gradient because its output
  only entered a zero-initialized context residual. A bounded direct sampled
  feature residual now opens that path without changing v3 checkpoints.
- `room_instance_mask_aware_adjacency=True` adds differentiable 2 px mask-contact
  logits to the query-similarity adjacency logits. The contact scale and mask
  geometry both receive gradients, so the consumer uses the same geometric
  relation as the canonical type-edge evaluator.
- `room_instance_count_aware_graph=True` runs the existing query-set head once
  as a draft before graph conditioning. Its soft training selection or
  candidate inference selection suppresses rejected slots before message
  aggregation; the final query-set pass still controls count, class counts, and
  selected masks.
- `typed_opening_host_gate=True` adds a one-parameter, identity-initialized log
  prior to unified opening logits. Its host support is predicted wall
  probability multiplied by junction/room-boundary evidence, creating a direct
  gradient while leaving v3 outputs unchanged at initialization.
- `refresh_validation_tokens=True` performs a fourth action-graph refresh after
  furniture. The validation stage therefore consumes final furniture evidence
  and matches the declared CAD layer order.

The v4 config adds 74,698 parameters (`6,880,597` versus `6,805,899`). On a real
predicted-control case, all output/loss tensors are finite and the adaptive
reference, contact scale, mask geometry, count-aware graph presence, contact
context, adaptive context, typed opening host-gate, and post-furniture
validation-refresh checks pass. The v3 checkpoint still loads under its original
defaults. Source:
`outputs/reports/predicted_room_dynamic_contact_audit_v1.json`.

This is a candidate, not a score. The next matched experiment must compare v3,
adaptive-only, contact-only, and joint arms on the same 512/64/64 screen. Only
the joint arm may proceed if it improves Instance F1 and Edge/GED without
lowering Node F1 or safety. Opening is a separate matched arm owned by the
host/center-width consumer.

## Wall Budget Dead-Path Repair

The CAD wall candidate scorer had a deterministic initialization defect:
global and exterior/interior budget heads zero-initialized both their hidden and
final projections. A direct first-step gradient audit showed the structural
weights received exactly zero gradient and only the biases trained. The
predicted wall budget therefore remained effectively sample-independent.

v266 keeps the original output biases but Xavier-initializes the hidden
projections and gives the final projections a tiny non-zero initialization.
Global, exterior, and interior heads now receive non-zero gradients at both
layers, and independent samples produce different budgets. Source:
`outputs/reports/cad_wall_budget_path_audit_v1.json`.

This is a wall-consumer path repair. Existing wall checkpoints still contain the
old zero weights and must be retrained on the source-aware skeleton portfolio
before any Wall F1 or precision gain is claimed.

## Furniture Active-Query Consumer

The furniture query head already emitted `query_budget` and `active_queries`,
but hard rendering, differentiable rendering, training evaluation, the
checkpoint bridge, and the diagnostic renderer ignored the active mask. The
canonical furniture layer therefore still selected the highest-scoring result
from all 32 queries, making the learned dynamic budget inert.

Both renderers now apply the active-query mask before max projection. A
deterministic two-query audit confirms that an inactive higher-scoring query
cannot enter the rendered layer, while the selected active query remains
visible. The CAD vector readout also consumes the same active mask; `query_budget`
remains only as a compatibility fallback, and disagreement fails closed. All
production raster and vector call sites pass the active mask. Source:
`outputs/reports/furniture_query_budget_consumer_audit_v1.json`.

This is a consumer repair, not a furniture-score result. Matched furniture
screening is still required before making an F1, type, assignment, or
containment claim.

## Predicted-Edit Ghost State

The count gate previously masked inactive room tokens, classes, existence, and
adjacency, but synthesized room instance IDs for every slot. Program assignment,
query masks, and auxiliary query masks also retained ghost values.

v269 binds all query-indexed edit tensors to one active mask. Inactive instance
IDs become `-1`; masks, auxiliary masks, program assignments, and scheduled
assignments are saturated; active IDs remain unchanged. Source:
`outputs/reports/predicted_edit_ghost_state_audit_v1.json`.

This strengthens the typed edit-state contract. It does not establish predicted
editing quality or strict-locality metrics.

## Revision Gate Consumer

`RevisionStateHead` emitted `revision_gate`, but the learned-revision consumer
ignored it. Low-confidence learned family and region predictions could
therefore expand a typed edit graph without admission control.

v270 consumes the gate. Admitted revisions may expand learned families and
regions; rejected revisions fall back to the declared edit region and infer
families only from actual changed labels. A deterministic high/low gate audit
confirms that the learned wall extension appears only in the admitted arm while
the fallback preserves the room-only anchor. Source:
`outputs/reports/revision_gate_consumer_audit_v1.json`.

v271 additionally supervises the gate from whether the target edit region is
nonempty. The revision objective now reports `revision_gate_loss`, and the audit
confirms a nonzero first-step gradient into the gate head. This is edit-routing
consumer and optimization evidence, not an editing-quality result.

## Predicted Object-Delete Execution

The predicted instruction projector previously returned unchanged labels for
`opening_delete` and `furniture_delete`, while the evaluator still marked those
operations as executed. This made both categories silent no-ops.

v272 deletes only the requested semantic family inside the declared region,
preserves room pixels outside that family, and verifies exact equality outside
the edit mask. Both opening and furniture deletion audits pass. Source:
`outputs/reports/predicted_edit_object_delete_audit_v1.json`.

This is execution-contract evidence. It does not yet claim matched predicted
edit quality.

The repaired executor was run through the canonical CPU evaluator on 40 typed
prompts: five examples from each executed operation and each abstention
category. All 25 executed prompts preserve outside pixel/room exactness and
validity at `1.0`; all 15 abstentions preserve the input exactly. Node F1 rises
by `+0.070973` with CI `[0.041930, 0.120634]`.

The overall gate remains failed: Edge overlap is exactly unchanged, and GED
moves by `-2.08` with a zero-crossing CI `[-5.214, 1.924]` while 18 of 25 cases
worsen. Locality and execution are fixed; graph quality is the remaining edit
blocker. Source:
`outputs/reports/residential_predicted_edit_objectdeletefix_smoke5_v1.json`.

The next same-case run preserved the full multilabel edited room partition
instead of collapsing changed pixels to one mode class. Node F1 improves by
`+0.072166` with CI `[0.041717, 0.119674]`, while GED changes by `-0.64` with
CI `[-4.855, 4.082]`; 8 cases improve and 16 worsen. Edge overlap remains
unchanged, so the overall gate still fails. Source:
`outputs/reports/residential_predicted_edit_multilabel_smoke5_v2.json`.

The original Edge gate required a positive paired delta even when the baseline
was already `1.0`, so it could never pass. v275 allows exact non-regression at a
declared ceiling and adds a strict paired-improvement gate for GED. Re-evaluating
the same smoke now marks Node, Edge-ceiling, locality, validity, and abstention
as passing; GED remains the sole failing check with CI `[-4.855, 4.082]`.
Source:
`outputs/reports/residential_predicted_edit_multilabel_gatefix_smoke5_v3.json`.

The target-free `tiled_purity_050` graph readout was then tested on the same
cases. It leaves every executed Node/Edge/GED value unchanged while degrading
abstention from exact `node_f1=1, ged=0` to `0.9867/1.0`, so the route is
rejected and `direct_tiled` remains the control. Source:
`outputs/reports/residential_predicted_edit_purityreadout_smoke5_v4.json`.

The smoke evaluator now reports GED decomposition, absolute Node/GED gates, and
matched-node coverage. Executed edits match only `0.04-0.15` of target room
nodes; absolute Node F1 is `0.04-0.10` and refined mean GED is `16-27`.
Conditional Edge overlap is therefore not a quality signal. Room addition is
dominated by `14.2` node deletions per case, while furniture deletion is
dominated by `4.6` substitutions and `12.2` edge insertions. Source:
`outputs/reports/residential_predicted_edit_absgate_smoke5_v6.json`.

The evaluator now isolates edit-induced graph change with same-case
pre-edit/post-edit pairs. Across 25 executed cases, Node F1 changes by
`-0.010645` with CI `[-0.030412, 0]`, Edge overlap is unchanged, and GED rises
by `+3.2` with CI `[0.396, 8.142]`. Room addition alone accounts for `+15.4`
GED and two Node regressions; room type change has only a `-0.2` GED trend with
a crossing interval. Source:
`outputs/reports/residential_predicted_edit_effect_smoke5_v7.json`.

Room add/type projection was then restricted to the dominant target room class:
it no longer copies every GT/pred semantic difference inside the region.
Room-addition Node F1 rises `0.03844 -> 0.07619`, GED falls `27.2 -> 24.6`,
and average node deletion falls `14.2 -> 6.8`. Overall edit effect improves to
Node `-0.003095` and GED `+2.8`, but both remain negative and the absolute
coverage/Node/GED gates still fail. Source:
`outputs/reports/residential_predicted_edit_dominant_target_smoke5_v8.json`.

Keeping only the largest connected component of the dominant target gives a
larger reduction: room-addition GED falls `24.6 -> 12.8`, average node deletion
falls `6.8 -> 1.8`, and graph-readout GED improves by `-3.36` with CI
`[-6.121, -0.037]`. Edit-induced GED improves to `+0.48` with CI
`[0.12, 1.041]`, while Node remains `-0.002694`. Absolute Node/GED and matched
coverage still fail. Source:
`outputs/reports/residential_predicted_edit_roomcomponent_smoke5_v9.json`.

Applying the same connected-target rule to room removal is rejected. Room GED
stays at `17.0`, room-deletion Node F1 falls `0.0667 -> 0.0571`, aggregate edit
Node worsens, and the significant graph-readout GED interval disappears. The
original neighbor-mode merge is restored. Source:
`outputs/reports/residential_predicted_edit_roomremovecomponent_smoke5_v10.json`.

Room execution is now explicitly operation-specific: room-add uses dominant
connected-target insertion, room-type uses direct edited-label conversion, and
room-remove keeps neighbor-mode merge. Room-type GED returns `16.6 -> 16.0`.
Aggregate edit GED falls `+0.48 -> +0.36` with CI `[-0.04, 0.961]`, while
graph-readout GED improves by `-3.48` with CI `[-6.124, -0.158]`. Absolute
Node/GED and matched coverage still fail. Source:
`outputs/reports/residential_predicted_edit_roomexecsplit_smoke5_v11.json`.

## Unified Type-Graph Edit Metrics

The predicted-edit evaluator previously called its instance room graph
`node_f1/edge_overlap/ged`, while the main HouseMind-style comparison used a
unique room-type graph. v285 separates the scopes. Canonical edit gates now use
`housemind_style_metrics`; instance wall/direct graph readouts stay under
`instance_*` diagnostics and are never pooled with type metrics.

On the same 40 prompts, executed categories reach type-node F1 `0.4256-0.6223`,
type-edge overlap `0.0806-0.1210`, and type GED `7.8-21.0`. Edit-induced type
change is Node `-0.005628`, Edge `-0.001071`, and GED `0.0`. The absolute
Node/Edge/GED gates still fail, but the edit table is now semantically aligned
with the main external comparison. Source:
`outputs/reports/residential_predicted_edit_typegraph_smoke5_v12.json`.

## Gate-Balanced Checkpoint Selection

The prior selection score gave Room/Instance terms the largest weight while
Opening, Wall, and graph metrics were weak add-ons. `selection_score` now uses
equal-weight progress toward the declared gates for room macro IoU, instance F1,
type accuracy, wall, opening, furniture, node, edge, count, and GED. Values below
a gate retain negative progress, preventing a slight Node movement from hiding a
larger GED/count regression.

Re-ranking 42 training histories with at least three comparable gate metrics
changes 13 best epochs. The old aggregate selector therefore selected a
non-gate-optimal checkpoint in a material fraction of runs. Historical reports
and files are left unchanged, but future training must use the gate-balanced
selector and old selected checkpoints are not automatically authoritative under
the new policy. Source:
`outputs/reports/gate_balanced_checkpoint_selection_audit_v1.json`.

Two promoted-history checks explain the policy change. On
`typed_channel_attention_screen256_v1`, the new selector chooses epoch 4 over
epoch 5 because Count MAE improves `3.03125 -> 2.78125` across its gate despite
a small Room-IoU tradeoff. On `typed_channel_attention_highres_screen256_v1`,
epoch 2 replaces epoch 4 because both fail count similarly and epoch 2 has
higher Room IoU and type accuracy.

Checkpoint retention is insufficient to repair the 13 historical selections:
all 13 gate-selected replacement epochs are unavailable, and 2 old selected
epochs are also absent. The remaining 11 old selections can be recovered only
through `best.pt`, not the newly preferred epoch. Those branches must be
retrained or rerun under the gate-balanced selector; no offline re-selection is
claimed.

RPLAN uses the same principle through `rplan_selection_score`: room IoU,
instance F1, type accuracy, count MAE, explicit contract validity, token
accuracy/F1/CE/grammar, generation completion, and token-consumer
validity/watertight regressions all contribute normalized progress. Wall and
opening losses plus room-adjacency BCE are also first-class progress terms. A
one-case closed-vocabulary CPU contract smoke completes forward, backward,
validation, and report generation with `selection_score=-0.1113326`; this
correctly exposes weak wall/opening/graph behavior instead of allowing room and
token terms to hide it. This is integration evidence only and leaves the test
split closed. Source:
`outputs/train/rplan_gatebalanced_cpu_smoke_v2/report.json`.

## Current External Availability

The HouseMind project page was reachable on `2026-11-02T01:39:37Z`; it exposes
an online demo and weights, but still labels code as `Code (Soon)`. A GitHub
repository search for `HouseMind floorplan` returned zero public repositories.
The complete tokenizer, prompt/decoding contract, and VQ-VAE post-processing
therefore remain unavailable for a matched editing score.

CE2EPlan has a public repository (`main`, last push `2026-03-16T15:46:26Z`) but
no detected license and remains generation-only. Source:
`outputs/reports/external_availability_current_v1.json`.

## Frozen Package Audit

The current `cvpr_package_fp_v17` is internally consistent: package directory,
manifest, and archive contain the same `15,887` files; every packaged artifact
hash is current; sidecar and independent archive rebuild both match the current
archive SHA-256 `9fc00ce4586c1d82a01cc7b5fdcab4f804e31595cc21434a03301cbca8a9a02d`.

This does not match the historical expected fingerprint `3c0738f7...c3df`.
The discrepancy is therefore recorded as post-freeze package drift rather than
silently rewriting the expected checksum. Source:
`outputs/reports/cvpr_risk_closure_audit_v1.json`.

The v18 FEVP-2 vector package succeeds v17 as the current authoritative evidence
package. Its package, manifest, and archive all contain `15,979` files with 509
artifacts; sidecar and independent deterministic rebuild match
`907e4b42d94a58b6ddd335941b3bd1745fecb72304278b7d3fa95bf82b8b2403`.
Claim boundaries are present and no HouseMind defeat phrase is found. Source:
`outputs/reports/cvpr_vector_package_audit_v1.json`.

## Claim Boundary

These repairs answer why the current architecture could not optimize the weak
metrics, but they are not metric evidence. The next GPU round must compare the
same cases, seed, parameter count, and exposure against the pre-repair control.
Only a calibration plus disjoint-holdout gain that survives the canonical
Room/Node/Edge/GED evaluators can enter a publication table.
