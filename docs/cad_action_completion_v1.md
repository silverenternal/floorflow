# CAD Action Completion v1

Date: 2026-11-01

The five previously open CAD-action links now have explicit implementations.

| Gap | Implemented path | Consumer |
|---|---|---|
| Program to room instances | `ProgramInstanceRouter` plus Hungarian/schedule loss | room class/presence |
| Typed junctions | endpoint clustering into coordinates, degree, and incident walls | candidate plan JSON |
| Generate-validate-repair | verifier-gated accept/rollback state loop | typed repair |
| Furniture semantics | room-context semantic residual | furniture query classes |
| Learned revision state | operation/family/region head | typed edit graph |

## Program to Room Instances

The room instance stage now consumes the program schedule and predicts an
explicit query-to-room-class assignment. Supervision combines mask-IoU
Hungarian matching, foreground class cross entropy, and a differentiable
schedule-presence loss.

## Typed Junctions

Selected wall endpoints are clustered into explicit junction vectors. Each
junction records a coordinate, degree, and incident wall IDs. Candidate plan
JSON now serializes these junctions instead of leaving them implicit.

## Verify and Repair

The repair loop proposes one typed edit at a time. A verifier must approve the
candidate before it becomes the new state. Rejected or missing edits keep the
previous best state, preventing repair-chain drift.

## Furniture Semantics

Furniture query logits receive a room-schedule-conditioned residual. The model
can learn priors such as toilet/sink to bathroom and bed to bedroom while
retaining the existing geometry and containment path.

## Revision State

`RevisionStateHead` predicts revision operation, semantic family, gate, and
region from image state plus the edit mask. `consume_learned_revision` uses the
learned family and region to construct the typed edit graph.

## Predicted-Head Capacity

The source-router candidate with room-instance and furniture stages is about
6.14M parameters. The added branches are approximately:

| Branch | Parameters |
|---|---:|
| Program instance router | 23,066 |
| Room instance stage | 636,022 |
| Furniture query stage | 539,152 |
| Furniture semantic router | 25,562 |

The experiment contract and controls are in
`configs/cad_action_completion_v1.json`.
