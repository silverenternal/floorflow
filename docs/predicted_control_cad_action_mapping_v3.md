# Predicted-Control to CAD Design Action Map v3

Date: 2026-11-01

## Purpose

This audit maps each residential CAD production action to an explicit
predicted-control input, network state, and downstream consumer. A module name
without a consumer does not count as coverage.

## Production Sequence

The predicted-control protocol is interpreted as this production sequence:

```text
site/envelope
-> program and room schedule
-> zoning and room partition
-> room identity and cardinality
-> wall network
-> junction nodes and endpoint continuity
-> doors/windows on wall hosts
-> furniture and fixtures
-> dimensions/annotations
-> validation, repair, and revision
```

## Action-to-Network Map

| CAD action | Predicted-control signal | FloorFlow network state | Downstream consumer | Status |
|---|---|---|---|---|
| Set exterior envelope | footprint/support raster | learned envelope source token plus action envelope token | room, wall, room-boundary, and channel-source paths | explicit |
| Build program schedule | room-type program vector | learned program source token plus action program token | room and semantic-channel queries | explicit source, weak per-instance supervision |
| Zone and partition | room logits / support | room-instance decoder and query ownership | room cells and wall candidates | implemented |
| Assign room identity/count | room logits / program | query class, presence, and count state | room layer and graph nodes | implemented, absolute IoU limited |
| Generate wall network | structure and owner boundary | wall stage, structure bridge, candidate scorer | typed wall vectors | implemented |
| Create junction nodes | junction probability, wall endpoints | junction action token, candidate adjacency, degree head | endpoint graph and topology repair | partial |
| Place doors/windows | opening family map and wall hosts | opening stage, host interval router, candidate wall IDs | host-aware opening vectors | implemented, quality limited |
| Place furniture/fixtures | furniture field | dense furniture layer and optional query stage | room-legal furniture polygons | implemented, semantic quality limited |
| Add dimensions/annotations | none in this protocol | no architectural annotation stage | none | out of scope |
| Validate and repair | support, validity, watertightness | validation token and topology repair head | bounded plan repair | implemented, promotion open |
| Revise/edit | local-delta control and typed state | typed edit graph and active-stage state | strict-locality composition | implemented, revision state not fully learned |

## CAD Layer Map

| CAD layer | Network representation | Native output |
|---|---|---|
| Envelope/support | footprint and learned envelope token | support raster and boundary condition |
| Room | instance queries, owner map, class logits | labelled room polygons |
| Wall | staged wall map, centerline, candidate graph | typed wall segments |
| Junction | junction probability, endpoint adjacency, degree head | graph nodes/degrees |
| Opening | family map, wall host, interval and type | host-wall opening segments |
| Furniture | dense field and room-conditioned queries | room-clipped furniture polygons |
| Validation | validity, watertightness, constraint token | accept/repair diagnostics |
| Annotation | none | excluded from the claim |

## New Mappings Added in v3

1. The envelope is now an explicit learned source token rather than only an
   average-pooled action-graph input.
2. The room program is now a separate source that every semantic channel can
   attend to.
3. Semantic channel queries become sample-conditioned.
4. Pixel cross-attention is gated by channel query and attended context.
5. Weak wall/opening/furniture stages use stage-level MIta.
6. The CAD action graph has causal MIta history while preserving exact
   predecessor attention.
7. Opening hosts use hybrid dense+MIta routing over candidate walls.
8. The structure bridge uses multi-scale MIta over envelope, room, and wall
   evidence.

## Closed Mapping Gaps

The five model-mapping gaps from the previous audit are now implemented:

1. Junction endpoints are clustered into typed coordinate/degree/incidence
   vectors and serialized in the candidate plan.
2. Program-to-room-instance assignment now has mask-Hungarian supervision and
   a schedule-consistency objective.
3. Generate-validate-repair now runs as a verifier-gated state loop with
   rollback to the best accepted state.
4. Furniture semantics receive a room-schedule-conditioned residual while
   retaining geometry and containment.
5. Revision operations, semantic families, confidence, and regions are
   predicted by a learned revision-state head and consumed by the typed edit
   graph.

Architectural dimensions, labels, and hatching remain outside the declared
residential predicted-control protocol. Their absence is a scope boundary, not
an unmodeled step in the claim.

The implementation contract is recorded in
`docs/cad_action_completion_v1.md`.

## Required Execution Order

1. Source-router head screen.
2. Program-instance and furniture-semantic screen.
3. Opening host-MIta screen.
4. Structure bridge multi-scale screen.
5. Room query-set MIta screen.
6. Typed-junction and verification-loop integration screen.
7. Learned-revision screen against the deterministic typed graph.

This keeps action coverage and metric ownership attributable.
