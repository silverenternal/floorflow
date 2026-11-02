# S3 Group Splice Rejection v1

Date: 2026-11-01

## Hypothesis

The 4-view candidate pool has complementary trajectories, while whole-trajectory
selection cannot improve every case. For each positive cluster group, choose the
candidate whose decoded blocks have the fewest exact edge-contact components,
then splice that complete group into the selector-chosen trajectory. Candidate
states are never averaged and exact block areas are preserved.

## CPU Screen

On the frozen 92-case development split:

- the selector chooses base views `[41, 16, 17, 18]`;
- 203 of 325 cluster groups switch source view;
- raw grouping components fall from `630` to `429` (`-31.90%`);
- raw grouping violations fall from `305` to `104` (`-65.90%`);
- 76 cases improve, 16 are unchanged and none worsen.

This is a strong pre-legalization mechanism signal.

## Legalized Result

The mixed trajectory is projected and evaluated with the same exact-area
projector and `scale_pack` legalizer as every control:

| Metric | Group splice | Selector base | Relative |
| --- | ---: | ---: | ---: |
| Cost | 1.212734 | 1.101557 | +10.09% |
| V_rel | 0.041054 | 0.029114 | +41.01% |
| Area gap | 0.116742 | 0.047681 | +144.84% |
| b2b HPWL gap | 0.072410 | 0.018112 | +299.79% |
| p2b HPWL gap | 0.158597 | 0.004126 | +3744.12% |
| Grouping | 1.054348 | 0.739130 | +42.65% |
| Boundary | 1.076087 | 0.782609 | +37.50% |
| Rho MAE | 0.262195 | 0.235030 | +11.56% |
| Legalizer rung | 1.989130 | 1.684783 | +18.06% |

Only 31 of 92 cases preserve the joint cost/V_rel comparison after
legalization.

## Decision

Reject pre-legalization group splicing. It verifies the S2 diagnosis: the
dominant bottleneck is legalization survival, not raw trajectory selection.
Do not add this arm to the selector. The next group-state repair must operate
in a legalizer-compatible coordinate or feasibility space and must optimize
rung, displacement and post-legalized grouping directly.

## Rigid Group-Frame Follow-up

An additional 92-case screen translated each selected cluster to the base
trajectory's cluster center, removing the absolute-coordinate mismatch. This
made legalization worse:

- cost `+442.67%`;
- V_rel `+405.78%`;
- grouping `+366.18%`;
- legalizer rung `+41.94%`;
- only 8/92 joint wins.

Rigid relocation changes overlap pressure around otherwise valid layouts and
forces the global packer to reflow almost everything. Both absolute-coordinate
and group-local pre-legalization splices are therefore closed.

