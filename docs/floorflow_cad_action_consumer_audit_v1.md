# FloorFlow CAD Action Consumer Audit v1

## Scope

This audit checks whether the existing `CadActionGraph` and vector decoder form a
real ordered CAD pipeline or only add stage tokens as auxiliary memory. It does
not open RPLAN test and does not modify chips/FloorSet code.

## Current Path

`UnifiedFloorplanGenerator` currently performs:

```text
boundary segments
  -> room identity/mask partition
  -> CadActionGraph stage tokens
  -> vector room/wall/opening decoder
  -> canonical owner-cell readout
```

The stage graph is a finite one-pass sequence with:

```text
envelope -> program -> rooms -> walls -> junctions
         -> openings -> furniture -> validation
```

Each stage only attends to declared predecessors. Stage gates and routed tokens
are returned in the model output.

## Consumers That Exist

- `rooms` stage token is added to routed room queries.
- `walls` queries consume room decoder state because `stage_context=True`.
- `openings` queries consume wall decoder state.
- Stage tokens enter the vector decoder memory.
- The canonical readout consumes the owner cell complex for room adjacency,
  shared-boundary walls, junctions, and the entrance host.

## Integration Gaps

1. All stage tokens are currently visible in one shared vector-decoder memory.
   The decoder does not apply a dedicated stage FiLM/context to wall and opening
   queries.
2. Learned wall outputs (`wall_exists`, endpoints, thickness, exterior logits)
   are not consumed by the promoted canonical `shared_network` wall path.
3. Learned opening outputs are not consumed by the promoted canonical readout;
   canonical openings remain condition-derived.
4. Stage exit is implicit in the single forward pass. There are no explicit
   per-stage exit states or operation-log records.
5. The `junctions`, `furniture`, and `validation` stage tokens have no direct
   trained canonical consumer in the RPLAN promotion path.

These gaps prevent calling the current path a complete learned CAD-action
pipeline. The owner cell complex is real, but wall/opening learning is not yet
closed over it.

## Required Closure

The next candidate must:

- add an opt-in stage conditioning module that leaves old checkpoints unchanged;
- make wall and opening decoders consume only their declared predecessor state;
- use learned wall evidence as a bounded refinement of owner-network walls;
- attach learned openings to the fused shared-wall graph;
- emit explicit stage completion states and operation logs;
- prove every stage has a downstream consumer before screening quality.

## Acceptance

- One forward pass returns a valid complete plan.
- Current checkpoints load byte-for-byte through the disabled branch.
- The candidate branch has finite gradients on a real batch.
- Room IoU/F1 regress by no more than 0.01.
- Wall F1 or wall IoU improves.
- Watertightness remains at least 0.998 in the mechanism screen and dangling
  endpoint rate remains at most 0.02.
- The candidate is rejected if a learned stage has no measurable consumer.

## Evidence Boundary

This is an internal architecture audit. It is not a paper draft, external
leaderboard result, or claim that learned openings already outperform the
condition-derived readout.
