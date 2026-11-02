# Predicted-Control CAD Action and Layer Map v2

## Purpose

This map checks each real residential CAD production action against an explicit
FloorFlow state and downstream consumer. A name appearing only in a loss or an
offline evaluator does not count as a model mapping.

## Production Action Map

| CAD production action | FloorFlow network state | Downstream consumer | Status |
| --- | --- | --- | --- |
| Set exterior envelope / footprint | footprint and support controls; envelope stage token | room completion, candidate footprint walls | Explicit input; envelope is not a learned stage |
| Build room program / schedule | program tensor; program stage token | room decoder context | Flat pooled evidence; no program-to-room token routing |
| Partition rooms and assign identities | raster-instance room stage; room logits; room cells | candidate boundary generation | Implemented; strongest current path |
| Generate wall axes / segments | staged wall map; room/footprint wall candidates; candidate graph scorer | candidate graph and final wall layer | Implemented; descriptor-first graph decoder is under screen |
| Create junctions and snap wall endpoints | geometry-head junction probability can enter the ordered CAD action graph; endpoint adjacency enters candidate graph attention | descriptor loss, graph attention, endpoint selection | Partially enabled: graph adjacency is active; junction-stage injection is opt-in until the layer model is retrained under the new contract |
| Classify wall layer / type | exterior/interior phase mask and separate phase budgets | exterior vs interior candidate top-K selection and `ReadoutWall.wall_type` | Implemented for candidate reconstruction |
| Place doors/windows on host walls | dedicated opening heatmap; high-recall host-wall skeleton; new many-query `OpeningHostRouter`; candidate wall host projection | opening layer via `routed_opening_rows` or detector fallback | Typed joint-host contract implemented; training against predicted opening tokens remains |
| Place furniture / fixtures | furniture stage logits; optional program/action-conditioned dynamic query stage | raster layer or room-attached vector furniture polygons | Typed vector contract implemented; query localization rejected at 64/32 |
| Dimensions, labels, hatching | no architectural annotation stage | none | Not modeled; should stay outside current residential SOTA claim |
| Validate and repair plan | validity/watertight/fpeval; validation stage token; bounded typed repair head | deterministic legalization and candidate-plan composition | Typed repair contract implemented; default budget rejected, two-edit arm statistically inconclusive |

## CAD Layer Map

| CAD-like layer | Model representation | Independent output | Current final-plan consumer |
| --- | --- | --- | --- |
| Support / envelope | footprint/support controls, support logits | yes | room partition and candidate footprint |
| Room | room instance stage and room logits | yes | explicit room cells |
| Wall | wall heatmap, centerline, candidate graph | yes | vector wall candidates |
| Junction / wall node | junction heatmap and candidate endpoint graph | probability and adjacency | graph attention and topological descriptors |
| Opening | door/window heatmap and family/type probability | yes | host-wall intervals |
| Furniture | furniture logits plus optional instance queries | yes | raster composition and room-attached vector polygons |
| Annotation / dimensions | none | no | none |
| Validation | audit state, constraint token, bounded repair actions | metrics and gates | learned repair head consumes final topology diagnostics |

## New Explicit Mappings

The current revision adds:

1. An opt-in `junction_probability` contract from the frozen geometry head to the
   CAD action graph. It is disabled for old checkpoints because they were
   trained while the junction token reused wall probability.
2. Shared-junction candidate adjacency inside the wall scorer.
3. Exterior/interior wall phase masks, separate budgets, and final wall type.
4. A network-predicted multiscale filtration descriptor that conditions
   candidate scores before selection.
5. An opt-in program/action-conditioned furniture query stage whose instances
   are vectorized as room-attached polygons and serialized in the CAD plan.
6. An identity-initialized bounded repair head that consumes the validation
   token and typed wall/junction/opening diagnostics.

## Remaining Structural Gaps

1. Junction nodes are still inferred from candidate endpoints rather than
   emitted as typed vectors.
2. The junction-stage injection needs a matched layer-model retrain before it
   can be enabled in production or evaluation.
3. Program controls are pooled, not routed by room role; an optional
   deterministic program bias exists but is not yet validated.
4. Opening host assignment is not jointly supervised with final selected walls.
5. Furniture now has instance-query and raster-fallback vectorization with a
   plan/CAD JSON consumer. The first matched screen found diffuse masks and
   rejected query promotion; keep the dense training path while redesigning
   localization.
6. Learned validation/repair now consumes wall, junction, opening, validity,
   and validation-token diagnostics and emits bounded graph edits, but it has
   not yet been trained or promoted.

The new opening router can assign multiple independent opening queries to the
same wall, unlike the legacy one-interval-per-wall head. Its outputs are already
consumed by `compose_scored_candidate_plan`; the remaining work is supervised
training and matched evaluation under the predicted-control protocol.

The first matched 64/32 eight-epoch router screen produced routed Opening F1
`0.0852` versus detector fallback `0.0868`, with validity `1.0` at the selected
checkpoint. Teacher-forced host supervision improved host accuracy during
training, but did not exceed the fallback at this data budget. Keep the
detector fallback as the production path and the router as a typed candidate
mechanism until a larger matched screen is justified.

These gaps define the next architecture tasks; they must not be described as
already solved.

## Audit Order

1. Retrain the layer stage under an explicit junction input only after a matched
   64/32 contract comparison.
2. Add a joint wall-host opening stage so host identity is supervised with final
   selected wall IDs rather than only projected during inference.
3. Validate program-conditioned room logits on the same split before enabling
   the optional program bias.
4. Redesign furniture queries with room-local mask attention and a small-object
   loss before repeating the matched screen.
5. Align bounded-repair training with composed endpoint improvement before any
   production enablement.

## PC-06M Screen Outcome

The 64/32 eight-epoch mechanism screens are recorded in
`outputs/reports/cad_matched_mechanism_screens_v1.json`.

- Furniture query with detached backbone preserved wall/opening behavior, but
  room-legal vectorization removed nearly all learned polygons because the query
  masks remained diffuse across rooms.
- Bounded repair at the default eight-edit budget over-edited 32 cases and
  reduced wall/opening F1. A two-edit validation sweep produced a safe but
  non-significant Wall F1 delta of `+0.00058` with a confidence interval crossing
  zero.
- Learned room-stage residuals lowered room macro IoU and opening F1; keep
  `preserve_room_stage=True`.

These mechanisms remain candidates/diagnostics, not main-claim evidence.

The follow-up PC-06N screen added small-object Tversky, room-containment
supervision, and per-room descriptor tokens. Mask IoU rose from `0.000455` to
`0.001257` and furniture F1 reached `0.16465`, but room-legal vector furniture
F1 remained `0.01809` and the selected checkpoint regressed opening F1 by
`0.03932`. The next required step is explicit query-to-room assignment and
room-local instance decoding, not more epochs on the room-token objective.

PC-06O added explicit query-to-room supervision and consumed the assignment in
the vectorizer. Furniture F1 reached `0.19422` and mask IoU `0.001392`, but
assignment accuracy was only `0.27604` against a `0.25651` all-background
baseline. Room-legal vector furniture F1 fell to `0.00541`, and opening F1
regressed by `0.02933`. The pixel-query family is now closed to further scaling;
furniture moves to direct room-conditioned vector-primitive supervision.

PC-06P/PC-06Q replaced furniture masks with room-slot vector primitives. Across
two 64/32 seeds, vector F1 averaged `0.21269` versus `0.04373` for the updated
dense vector control. At 256/64, vector F1 was `0.19577`, paired gain
`+0.16955`, polygon validity `0.96875`, attachment `0.48088`, containment
`1.0`, and out-of-boundary `0.0`. Matched type accuracy remained `0.15376`
and class balancing only reached `0.22225`, so furniture geometry is retained
as support evidence while semantic furniture SOTA is excluded from the main
claim.

