# FloorFlow CAD Data-Chain Audit v2

## Scope

This audit follows the deployed residential path:

```text
target-free image/control
  -> room identity and support
  -> room count and owner partition
  -> wall / junction / opening evidence
  -> canonical room cells and vector assembly
  -> target-comparative metrics
```

It checks data provenance, tensor contracts, model handoffs, and whether each
named CAD stage has a real downstream consumer. Unit-level stage-graph checks
are separate from evidence that the promoted model uses that graph.

## Production-Step Map

| CAD production step | Current mapping | Deployed consumer | Status |
| --- | --- | --- | --- |
| Set exterior envelope | footprint/control support | room labels and exterior composition | explicit input, not learned |
| Set program/schedule | program vector | optional room-class bias | flat pooling; not role-routed |
| Partition rooms | dense and query room branches | explicit room cells | strongest current stage |
| Decide room support | query masks and learned deletion | support-safe room labels | newly consumed and under promotion |
| Count instances | connected components and query evidence | room metric / vector rooms | over-prediction remains |
| Generate walls | structure bridge, centerline, owner boundaries | layered plan and candidates | independent evidence fusion |
| Generate junctions | centerline junction probability, endpoint graph | candidate attention / diagnostics | not a first-class promoted vector |
| Place openings | opening heatmap, structure channel, host router | room/wall plan | host attachment remains partly post-hoc |
| Place furniture | furniture controls/query heads | optional layer | not part of promoted predicted-CAD row |
| Validate and repair | validity/watertight/FEVP, typed repair | evaluator or bounded candidate | not a closed generation feedback loop |
| Edit/revision | typed edit graph and local delta | edit renderer | not in the predicted-control training loop |

The ordered `CadActionGraph` has correct directional predecessors and a passing
contract audit, but the strongest current predicted-control evaluator still
composes specialized room, structure, and opening predictors outside that graph.
The graph is therefore an architectural asset, not yet the deployed causal path.

## Findings

### 1. Support gating had no output consumer

Severity: critical, fixed.

The previous dense-query support gate trained a support objective, but the
renderer always multiplied room classes by the footprint. Support IoU therefore
never changed. The current selective-deletion head now emits a keep map consumed
by the room-label renderer.

### 2. Topology safety ran before footprint masking

Severity: critical, fixed.

An interim topology-safe renderer fed full-frame room classes into component
checks before applying the footprint. Full-frame background components were
restored as valid support, producing false rooms and invalid counts. The
consumer now enforces `class map -> footprint -> topology-safe keep -> labels`,
with a regression test.

### 3. Stage runtime mixed a 224px contract with implicit settings

Severity: high, fixed.

`CadStageRuntime` hardcoded room logits to 224px, defaulted to a 0.75 blend that
was superseded by the 0.5 candidate, and did not validate cached program or
structure channels. It now exposes `room_size`, defaults to blend 0.5, and fails
closed on taxonomy/channel drift before inference.

### 4. Stage tokens discard spatial geometry

Severity: high, open.

`CadActionGraph._token` adaptive-average-pools every spatial map. This is
reasonable for envelope/program state, but wall, junction, and opening stages
lose line endpoints, host adjacency, and room-local location. A future spatial
token contract must include a consumed map or set-of-primitives output, not only
add pooled channels.

### 5. Opening placement is not jointly hosted

Severity: high, open.

Opening prediction, host-wall selection, and final wall selection currently meet
through projection/fallback logic. The formal CAD order requires the opening
query to consume a selected host wall and predict an interval along that wall.
The existing `OpeningHostRouter` is the correct reuse point, but its prior screen
did not beat detector fallback.

### 6. Walls and junctions are separate evidence streams

Severity: high, open.

The structure bridge, centerline head, and owner boundary candidates are fused
after prediction. Junction probability is not a first-class vector state in the
promoted plan. Wall continuity can improve while opening hosts or graph
adjacency regress because there is no shared junction/host state.

### 7. The current controls are mostly raster-coarse

Severity: high, open.

The control cache carries room, structure, furniture, and support rasters at
different resolutions. It does not carry ordered wall, junction, or host-wall
tokens. A complete CAD-stage model requires a canonical vector target cache and
stage-specific masks, not another raster class channel.

### 8. Dynamic behavior is still concentrated in post-hoc thresholds

Severity: medium, partially addressed.

The selective support head is learned, but preservation confidence, keep
threshold, minimum deletable component area, wall thresholds, and opening host
distance remain fixed. Dynamic confidence should be consumed through typed
abstention or budget heads; changing thresholds per evaluation sample would not
be auditable.

### 9. Furniture and validation have no promoted predicted-CAD consumer

Severity: medium, open.

Query furniture and bounded repair exist, but their matched screens did not
clear the gate. They remain engineering options and must not be described as
part of the current learned CAD SOTA path.

## Repairs Completed in This Audit

- independent dense/query checkpoint adapter;
- complete structure consumer for the room-identity candidate;
- explicit query-support probability and label consumer;
- topology-safe deletion with connected-component non-increase;
- footprint-before-topology rendering contract and regression test;
- per-artifact SHA-256 and five-seed no-pooling aggregation;
- stage-control taxonomy validation and configurable stage resolution;
- default stage blend aligned with the current 0.5 candidate.

Evidence:

- `src/floorflow/heads/dense_query_support_gate.py`
- `src/floorflow/inference/cad_stage_runtime.py`
- `src/floorflow/heads/raster_control_bank.py`
- `outputs/reports/cad_action_layer_contract_audit_v3.json`
- `outputs/reports/dense_query_blend_fiveseed_v1.json`

## Next Architecture Priority

1. Promote the topology-safe selective support head through scale and complete
   structural gates.
2. Connect the accepted room/support state to `CadStageRuntime` as an explicit
   stage handoff rather than recomputing a fixed blend.
3. Add spatial wall/junction state with a named downstream vector consumer.
4. Retrain opening host selection against final selected wall IDs.
5. Build canonical vector control targets before attempting another global
   staged generator.

