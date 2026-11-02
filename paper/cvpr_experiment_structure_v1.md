# FloorFlow CVPR Experiment Structure

Status date: 2026-11-02

This is the current experiment and paper-structure note. The canonical protocol
summary is `docs/fp_oracle_control_protocol_status_v1.md`.

## Locked Claim

Under the HouseMind edit-detection protocol, control-gated FloorFlow reaches
Delta-IoU `0.956703-0.999801` and Delta-MSE `0.000019-0.000096` across five
CubiCasa test503 operations and five repeated seeds. It preserves exact outside
pixels, rooms, polygons, walls, openings, and unaffected edges, maintains
FP-Eval validity `1.0`, and reports conditional watertightness
`0.9579-0.9675`.

The published HouseMind-E anchors are Delta-IoU `0.608` and Delta-MSE `0.0019`.
FloorFlow exceeds both numeric anchors for every operation, but the comparison is
not a same-dataset leaderboard result because HouseMind reports RPLAN
natural-language editing and FloorFlow reports CubiCasa oracle controls.

This is a protocol-level positive result. It does not establish complete-method
SOTA or deployable predicted/free-text editing.

## Main Tables

1. `oracle_control_cvpr_dossier_v1.md`
   - compact headline, matched controls, mechanism audit, slices, and source map
2. `residential_sota_main_v5.md`
   - per-operation Delta-IoU and Delta-MSE
   - exact outside preservation, FP-Eval validity, FEVP validity, and watertightness
   - HouseMind-E protocol context and matched local baselines
3. `residential_edit_fiveseed_v1.md`
   - operation-level five-seed stability
   - matched Raster2Seq and paired significance diagnostics
4. `residential_graph_protocol_dominance_v1.md`
   - aligned HouseMind-style, canonical, and typed graph protocols
   - explicit statement that protocols are not pooled
5. `fp_complexity_strata.md`
   - room-count, interface, and edit-area behavior

## Appendix

1. `fevp_hardened_main_results_v2.md`
   - five-layer outside exactness, violation rate, paired statistics, and
     fixed-evaluator graph metrics
2. `fevp_protocol_attacks_v2.md`
   - deterministic evaluator-sensitivity attacks
3. `fp_protocol_boundary.md`
   - low-room-count, weak-interface, and low-signal watertightness boundaries
4. `fp_delta_mse_decomposition.md`
   - wall/opening boundary, room boundary, edit-interior, and outside errors
5. `fevp_coverage_matrix_v1.md`
   - external-method audit coverage and required N/A semantics

Seed-level and earlier tables such as `fp_main_results_v2.md` and
`fp_control_gated_main_results.md` are retained as provenance. The current
per-operation table is `residential_sota_main_v5.md`;
`fp_housemind_protocol_comparison.md` is the updated secondary context table.

## Narrative

The experiment section should establish:

1. Edit fidelity and locality are distinct axes.
2. Declaring an outside-freeze wrapper alone is insufficient: the matched
   Raster2Seq wrapper reaches only Delta-IoU `0.035690-0.164930`.
3. An outside-frozen editing-aware control reaches
   `0.731151-0.998508`, while FloorFlow reaches
   `0.956703-0.999801` and preserves exact outside content.
4. The dual-metric advantage appears on every typed operation and every
   repeated seed.
5. The primary local-delta metrics are deterministic on fixed rows; repeated
   seeds document reproducibility, not population-level variance.
6. FP-Eval validity is `1.0`, while the stricter independent FEVP five-check
   validity is reported separately as `0.9124-0.9569`.
7. Structural Node F1 `0.982250`, edge overlap `0.929475`, and GED `0.869544`
   remain below HouseMind-E's published `0.998`, `0.934`, and `0.467`.
8. Remaining limitations are target-derived controls, predicted/free-text
   deployment, missing move-wall/resize evidence, graph quality, and the lack of
   executable HouseMind inference.

## Required Boundary Text

- Raster2Seq is a matched reconstruction baseline, not a native editing model.
- The editing-aware entry is an outside-frozen local-redraw wrapper.
- HouseMind official inference remains unavailable.
- HouseMind-E values are metric context, not a same-dataset ranking.
- Oracle-control results are non-deployable.
- The result covers CubiCasa test503 and does not directly extrapolate to RPLAN.
- Delta-IoU and Delta-MSE use aligned definitions but different data and inputs.
- Aligned, canonical-instance, and typed graph protocols are never pooled.
