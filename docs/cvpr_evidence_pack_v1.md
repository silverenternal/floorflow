# CVPR evidence pack v1

Status: historical style-generation evidence pack. Current submission status is
`docs/cvpr_experiment_status_20261102.md`, and the current protocol evidence is
`docs/fp_oracle_control_protocol_status_v1.md`.

## Claim

We study contour-conditioned, style-controllable constrained floorplan
generation. A train-only style bank and an explicit deployment-side style token
control a shared MiTA-CFM backbone that generates hierarchical wall, opening,
room, furniture and support distributions. The task is one-to-many design
generation, not exact recovery of an unobserved reference interior.

## Method evidence

| Component | Evidence |
|---|---|
| Shared pixel/element backbone | `src/floorflow/distribution/cfm_model.py`, `hub.py` |
| Hierarchical cross-channel coupling | `element_head.py`, `pixel_head.py` |
| Source-initialized CFM | `sampling.py`, `docs/sampling_math.md` |
| Train-only style bank | `outputs/style_banks/cubicasa_train_v1.json`, `skv4_train_v2.json` |
| Style-controlled pixel/element/velocity residuals | `style_conditioning.py`, `style_semantic.py` |
| Leakage audit | `style_design_manifest_v1.json` |

## Main evidence

| Claim | Result | Evidence |
|---|---:|---|
| Legal footprint containment | 0 violating pixels on test503 | `outputs/eval/style_design_full_test503/report.json` |
| Room target recall | 0.99602 | `final_gated_targetfree_cad_test503_v1.json` |
| Room area distribution MAE | 0.03952 | same |
| Opening-wall alignment | 0.46073 | same |
| Furniture-room containment | 0.84095 | same |
| Conditional watertight | 0.98874 | same |
| Unconditional watertight | 0.92445 | same |
| CubiCasa all-style test coverage | worst 0.71257 | `distribution_style_prompt_r94_v2_all_styles_test503.json` |
| CubiCasa all-style room IoU | mean 0.10316, diagnostic only | same |
| CubiCasa gated style val8 | mean 0.11729, worst 0.11607 | `final_gated_style_stability_val8_v1.json` |
| CubiCasa frozen test503 | Room IoU 0.16752, instance IoU 0.51673 | `final_gated_targetfree_cad_test503_v1.json` |
| Overlay style separation | room-conditioned 0.18076 | `distribution_r94_overlay_style_separation_val32_w05.json` |
| skv4 calibrated style room IoU | mean 0.12317, diagnostic only | `distribution_skv4_style_v1_calibrated_test296.json` |
| Style separation | 0.08572 pixel disagreement | `distribution_style_prompt_r94_v2_separation_val32.json` |
| Local retrieval baseline | coverage 0.93750, density 24.28, room IoU 0.07615 | `style_retrieval_baseline_style3_val64.json` |
| Final gated target-free test503 | Room IoU 0.16752, balanced CAD 0.33198 | `final_gated_targetfree_cad_test503_v1.json` |
| Adversarial self-control objects | opening/furniture/support IoU 0.20481/0.12627/0.96923 | `adversarial_self_control_refiners_test503_v1.json` |

## External baseline readiness

Official HouseDiffusion, House-GAN++, and MaskPLAN checkpoints are present and
strict-load successfully. House-GAN++ has a CubiCasa val64 oracle-program table;
HouseDiffusion converts 58/64 cases and passes a finite forward; MaskPLAN has a
64-case native RPLAN boundary-only table. See `docs/external_baselines.md` and
`outputs/reports/external_baseline_availability_v1.json`.

## Scope boundaries

- Do not claim exact contour-only interior reconstruction.
- Do not claim official HouseDiffusion, House-GAN++ or MaskPLAN numbers until
  their CubiCasa adapters are frozen and the comparable split is evaluated.
- Do not use Reference IoU as a primary acceptance metric.
- Do not introduce human annotation, ratings, surveys or participant data.
- Do not mix target-derived coarse-control upper bounds into the main result.

## Missing evidence

1. A fully contour-only official external baseline under the same protocol;
   current official tables are oracle-program or native-RPLAN.
2. Strong object, opening and vector-readout metrics; only the legality and
   conditional-generation hard floors pass.
3. Second-domain overlay replication on skv4.

## Planned figures and tables

- Figure 1: contour + style token -> hierarchical multichannel generation.
- Figure 2: fixed-case qualitative panels with style swaps.
- Figure 3: validity-versus-reference-IoU trade-off.
- Table 1: CubiCasa and skv4 quality/coverage/density/MMD.
- Table 2: style separation and ablations.
- Table 3: external and local baseline comparison.
- Table 4: complete CAD metric hierarchy and hard-gate status.

