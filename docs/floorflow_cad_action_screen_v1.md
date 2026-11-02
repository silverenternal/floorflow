# FloorFlow Finite CAD-Action Screen v1

## Scope

This screen tests whether ordered wall/opening stage tokens provide a measurable
canonical CAD benefit when the room backbone is frozen. It is a 128/64
validation mechanism screen with seed 2905 and one NFE. RPLAN test remains
closed.

## Candidate

The candidate adds two zero-initialized stage conditioners:

- wall queries consume the `walls` action token;
- opening queries consume the averaged `junctions` and `openings` tokens.

The branch is opt-in. The existing checkpoint loads with zero missing or
unexpected tensors when disabled.

## Clean Isolation

The first warm-start attempt allowed BatchNorm running statistics to change
despite frozen parameters. That arm was rejected. The corrected run keeps the
entire model in eval mode and trains only `stage_conditioners`.

Audit of the resulting checkpoint:

- non-stage tensors checked: `422`;
- changed non-stage tensors: `0`;
- trainable parameters: `56,256`;
- total parameters: `13,697,702`;
- canonical token round trip smoke: exact;
- stage exit logs per plan: `9`;
- incomplete stage exits: `0`.

## Results

| Arm | Room IoU | Instance F1 | Node F1 | Edge overlap | GED | Wall F1 | Wall IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 0.50587 | 0.22805 | 0.84810 | 0.39979 | 4.60938 | 0.56923 | 0.37546 |
| Stage-conditioned | 0.50587 | 0.22805 | 0.84810 | 0.39979 | 4.60938 | 0.56923 | 0.37546 |
| Control + learned fusion | 0.50587 | 0.22805 | 0.84810 | 0.39979 | 4.60938 | 0.57058 | 0.37619 |
| Stage + learned fusion | 0.50587 | 0.22805 | 0.84810 | 0.39979 | 4.60938 | 0.57000 | 0.37595 |

All arms have validity `1.0`, conditional watertightness `1.0`, opening F1
`1.0`, and dangling endpoint rate about `0.0101`.

Learned-wall fusion alone gives a small positive wall result:
`+0.00135` wall F1 and `+0.00073` wall IoU on the control checkpoint. Training
the stage conditioners does not improve this result. Relative to the matched
control+fusion arm, the stage-conditioned+fusion arm changes wall F1 by
`-0.00058` and wall IoU by `-0.00023`.

## Decision

- Do not promote the FiLM stage conditioning branch.
- Do not continue sweeping the stage residual scale.
- Retain the opt-in implementation and stage-exit audit as diagnostic tools.
- Keep learned-wall endpoint fusion as a bounded diagnostic, not a main claim.
- Keep the owner-network wall path and the existing five-seed result unchanged.

The result changes the next architecture action: global stage FiLM is too weak
and indirect. The next candidate must reuse `CadWallCandidateScorer` as a
per-owner-edge action selector conditioned by the wall stage token, trained
against matched target edges, with bounded keep/residual outputs that preserve
the shared owner topology.

Evidence:
`outputs/reports/floorflow_cadaction_screen128_v1.json`.
