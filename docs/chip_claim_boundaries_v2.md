# Chip Claim Boundaries v2

## Promoted claim

Under the same CFM objective, data, schedule, and exact-area decoder, moving
graph conditioning from an invariant global slot to per-block graph-token
trajectories improves the three-seed mean official-100 refined endpoints.

This remains distinct from FloorSet or CAD-generation SOTA.

## Demoted metrics

- MIB V_rel has zero signal on the official panel. Do not claim improvement.
- Boundary V_rel has a positive mean but an interval crossing zero. Report trend
  only.
- p2b/port HPWL has a large mean gain but an interval crossing zero. Report it
  as a structural signal, not a stable advantage.
- High-b2b and high-pin strata show little benefit and remain limitations.
- HouseMind-E is not a chip-domain comparator.

## Mechanism framing

- Cost reduction is concentrated in the failure tail: p50 improvement is
  `2.59%`, while p90/p95 are about `14.4%`.
- V_rel gains are not monotonic through the extreme tail.
- Fewer NFE/refiner steps reach declared baseline quality, but the primary
  endpoint remains NFE4 plus 400 refiner steps.

## Remaining diagnostics

The Trinity d192 gap must be decomposed by component, and floorplan weak bins
must remain visible rather than averaged away.

