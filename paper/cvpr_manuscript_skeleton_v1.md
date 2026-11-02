# FloorFlow CVPR Scoped Manuscript Skeleton v2

Status date: 2026-11-02. Status: `completed_ready_for_user_submission`.

The official CVPR review template is used by `paper/main.tex` and
`paper/supplement.tex`. The compiled main paper is six pages including figures
and tables, below the eight-page limit. The supplement is two pages. References,
the reproducibility statement, the data/ethics statement, and the mandatory LLM
Usage Statement are present.

## Argument

The manuscript studies strictly local floorplan editing under an explicit
control interface. It separates five axes: outside preservation, execution
safety, structural plausibility, topology legality, and reference fidelity.
Oracle controls establish an editing upper bound but are not deployable.
Target-free predicted controls expose the operating point, including abstention,
raw topology failures, weak room identity, and mixed semantic realization.

## Section Map

1. Introduction: locality is distinct from global reconstruction accuracy.
2. Related work: HouseMind is cross-dataset context; Raster2Seq is a wrapper.
3. Protocol: old state, declared region, typed change, outside domain, and
   abstention conditions.
4. Method: control transition, typed-before repair, change gating, deterministic
   routing, and the retained completion model.
5. Oracle results: 2,479 edits, five operations, and five repeated seeds.
6. Predicted results: val491 plausibility, operation safety, and the separately
   labeled older val191 semantic diagnostic.
7. Matched controls and composition ablation: paired same-case diagnostics.
8. Limitations and ethics: held-out status, weak fidelity, and unsupported
   operations remain visible.
9. Reproducibility and LLM Usage Statement.

## Canonical Tables

- `main_oracle_anchor_v1`: per-operation oracle edit support, outside exactness,
  validity, and raw/functional topology.
- `main_predicted_reliability_v1`: val491 plausibility, 738-prompt safety,
  abstention, and val191 semantic diagnostics.
- `main_resource_and_ablation_v1`: same-case Raster2Seq and editing-aware
  controls, composition arms, and routing cost.
- `main_external_context_v1`: heterogeneous HouseMind-E, Raster2Seq, typed, and
  predicted-fidelity context without a leaderboard.

## Canonical Figures

- `floorplan_protocol_overview_v1`: outside domain and edit flow.
- `predicted_control_operation_panels_v1`: safety, plausibility, and semantic
  realization as separate panels.
- `floorplan_failure_taxonomy_v1`: supported-operation outcomes and one retained
  failed case selected by a recorded rule.
- `floorplan_qualitative_cases_v1`: one oracle positive and one predicted failure
  loaded from NPZ evidence.

## Frozen Headline Values

- Oracle Delta-IoU: `0.956703-0.999801`.
- Oracle Delta-MSE: `0.000019-0.000096`.
- Outside pixel, room, polygon, wall, opening, and unaffected-edge exactness:
  `1.0`.
- Raster2Seq wrapper Delta-IoU: `0.137160-0.189491`.
- Predicted raster plausibility: `0.984929`; plan plausibility: `0.986558`.
- Predicted FP-Eval validity: `1.0`; raw watertightness: `0.816701`;
  functional watertightness: `0.997963`.
- Operation safety: 438 supported prompts executed; 300 unsupported or ambiguous
  prompts abstained; outside pixel and room-label exactness `1.0`; validity `1.0`.
- Older val191 semantic success: room addition `92/100`, room type `55/100`, room
  removal `20/100`, opening deletion `30/59`, furniture deletion `0/15`.

## Locked Boundaries

No complete HouseMind or RPLAN defeat, full-CAD SOTA, deployable free-text
editing, high predicted room identity, uniform graph superiority, or fresh
held-out generalization is claimed. Repeated seeds are not independent plans.
Raw and functional watertightness remain separate. Development-exposed operation
prompts are not held out. The optional held-out extension is
`optional_generalization_not_evaluated`.
