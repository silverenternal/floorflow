# Floorplan Held-Out Availability v1

Status date: 2026-11-02

## Finding

No strictly fresh compatible CubiCasa edit split is available in this workspace.
All 503 test cases were included in the full predicted-completion inference run,
including the 303 cases absent from the current edit-prompt report. They cannot
be described as unopened model cases.

## Decision

The optional fresh held-out task `FP-SUB-06` is deferred and marked
`not_applicable_to_scoped_release`. No held-out metric is fabricated, estimated,
or copied from a different split.

The main submission is the declared scoped release:

- oracle-control editing on CubiCasa test503;
- target-free structural reasonableness on val491 with five repeated seeds;
- development-exposed operation safety and abstention on test503;
- separately labeled semantic and fidelity diagnostics;
- identity fallback and external context outside the editing claims.

## Future Extension

A future held-out claim requires a pre-access freeze of the checkpoint,
thresholds, prompt manifest, evaluator, and taxonomy, followed by one execution
and publication of every outcome. The optional extension must remain separate
from the existing-evidence tables.

## Machine-Readable Status

- `outputs/reports/floorplan_heldout_availability_v1.json`
- `outputs/reports/floorplan_evidence_freeze_audit_v2.json`
