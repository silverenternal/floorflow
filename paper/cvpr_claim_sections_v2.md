# CVPR Vector-Claim Sections v2

This file contains vector-specific claim sections for internal review.
It is not a complete manuscript.

Status: the vector/FEVP claim remains valid for its declared experiment, but the
current submission-facing claim language is `cvpr_claim_sections_v3.md`.

## Abstract Claim

Under the FEVP-2 vector edit protocol, FloorFlow provides a complete audited result set on 12,395 five-seed rows over 2,479 CubiCasa test503 edit cases: room-polygon, wall-support, opening-support, and edge outside exactness are 1.0 in every operation, with Locality Violation Area 0. Under the same scorer, five-seed validity is 0.926-0.972, Node F1 is 0.771-0.890, and Edge overlap is 0.554-0.753. Conditional watertightness on complex room-count, interface, and medium/large edit-area strata is 0.966-0.970. Against a matched Raster2Seq outside-frozen reconstruction wrapper, FloorFlow's paired improvements in outside exactness, validity, Node F1, and Edge overlap have positive 95% bootstrap intervals in every seed and Bonferroni-adjusted Wilcoxon p <= 2.02e-23 across 20 paired tests. Inference-time residual refinement, restricted to affected wall and opening proposals, keeps room-label, polygon, wall, opening, and edge outside exactness at 1.0 with Locality Violation Area 0, and reduces wall MAE by 2.80%-4.98% and opening center MAE by 30.38%-39.69% across operations and five seeds.

## Boundary Statements

- Raster2Seq is a matched reconstruction wrapper, not an editing-aware method.
- The HouseMind and RenoLayout complete methods are not reproduced.
- CE2EPlan, FMLM, and TLC-Plan remain generation-only context.
- The result is specific to CubiCasa test503 and the FEVP-2 vector protocol.
- Strict locality excludes the declared edit region and changed wall support.
- No natural-language understanding or human-preference claim is made.

## Related-Work Positioning

HouseMind and RenoLayout target unified multimodal reasoning and renovation breadth. HouseMind uses discrete room-instance tokens for understanding, generation, and editing. CE2EPlan performs end-to-end boundary-conditioned vector floorplan generation with topology- and geometry-enhanced components. FMLM and TLC-Plan target unified or hierarchical vector generation. Raster2Seq is a matched reconstruction baseline and not an editing-aware model. CE2EPlan is a generation-only target; no generation result is interpreted as an editing result. The contribution is protocol and evidence architecture. We do not claim that FloorFlow beats HouseMind. Missing metrics define an audit boundary; absence defines an audit boundary, not a performance ranking.
