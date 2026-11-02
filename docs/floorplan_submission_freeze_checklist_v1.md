# FloorPlan Submission Freeze Checklist v2

Status date: 2026-11-02. Local status: `passed_scoped_release`; package status:
`ready_for_user_submission`.

## Scope

- [x] Use the existing oracle test503, predicted val491, and development-exposed
  operation evidence.
- [x] Keep the optional fresh held-out extension as
  `optional_generalization_not_evaluated`.
- [x] Do not represent OpenReview enrollment, upload, or a venue receipt as a
  completed local action.

## Evidence Identity

- [x] Source registry: `docs/floorplan_submission_source_registry_v2.json`.
- [x] Provenance audit: `outputs/reports/floorplan_source_provenance_audit_v2.json`.
- [x] Freeze: `configs/floorplan_submission_evidence_freeze_v2.json`.
- [x] Freeze audit: `outputs/reports/floorplan_evidence_freeze_audit_v2.json`.
- [x] Four generated tables and four PDF/SVG figure pairs.
- [x] Figure case manifest: `outputs/reports/floorplan_figure_case_manifest_v1.json`.
- [x] Local integrity audit:
  `outputs/reports/floorplan_submission_integrity_v1.json`.

## Claim Integrity

- [x] Oracle, predicted, identity-fallback, generation, and free-text populations
  are never pooled.
- [x] Repeated seeds are executions, not independent plans.
- [x] Raw and functional/componentwise watertightness remain separate.
- [x] Supported execution, abstention, and object/ghost-state contracts are not
  semantic success.
- [x] The val191 semantic diagnostic is labeled by model age and split.
- [x] HouseMind-E is cross-dataset context and Raster2Seq is a reconstruction
  wrapper.
- [x] Development-exposed test prompts are not called held out.

## Official Format

- [x] Official CVPR template and bibliography style are stored under `paper/`.
- [x] Main and supplement are anonymous and use the review option.
- [x] Main paper is six pages including figures and tables, within the eight-page
  limit.
- [x] References and the mandatory LLM Usage Statement are present.
- [x] No overfull box, undefined reference, or missing-file error remains.

## Required Commands

```bash
uv run python scripts/build_floorplan_submission_evidence_v2.py
uv run python scripts/build_floorplan_submission_tables_v2.py
uv run python scripts/build_floorplan_submission_figures_v1.py
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex)
uv run python scripts/prepare_predicted_control_submission_freeze_v1.py \
  --mode existing_evidence_scoped \
  --output configs/floorplan_submission_evidence_freeze_v2.json
uv run python scripts/audit_floorplan_evidence_freeze_v2.py
uv run python scripts/build_floorplan_submission_artifact_v1.py
uv run python scripts/audit_floorplan_submission_integrity_v1.py
```

The scoped freeze records the current commit and dirty state for transparency but
does not require a clean research worktree. Unrelated research changes remain
outside the selected release identity set.

## Optional Held-Out Extension

This branch is not invoked. It requires a compatible unopened split and a frozen
candidate before access. Its absence blocks only a fresh-generalization claim and
does not block the scoped oracle, predicted, safety, or structural-reasonableness
claims.

## Local Result

- Scoped release audit: `passed_scoped_release`.
- Artifact checklist: `ready_for_user_submission`.
- Submission state: ready for the user to review and authorize external upload.
