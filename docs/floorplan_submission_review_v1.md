# Floorplan Submission Review v2

Status date: 2026-11-02. Editorial decision: `ready_for_user_submission` for the
declared scoped release.

## Decision

The manuscript now has a complete argument, an official-format PDF, generated
tables and figures, a source registry, a frozen evidence identity set, and an
anonymous artifact package. The remaining external steps are author eligibility,
OpenReview enrollment, review of the mandatory LLM Usage Statement, and upload.
Those actions cannot be represented as completed without an external receipt.

## Editor

The paper fits the vision and design-system audience because it treats floorplan
editing as a controlled state transition rather than global image synthesis. The
strongest contribution is the explicit separation of exact locality, operation
safety, structural plausibility, topology legality, and reference fidelity.

The previous review treated a fresh held-out split and a newer model as universal
blockers. That framing was too broad. The current submission narrows the claims to
existing oracle, val491, development-exposed safety, and older semantic
diagnostic populations. The held-out extension is optional and explicitly not
evaluated.

## Methodology Reviewer

Resolved issues:

- every main-table number resolves to a registry pointer or a generated source;
- repeated seeds are described as repeated executions, not independent plans;
- paired controls use floorplan-level units and Holm adjustment;
- raw and functional watertightness are separate;
- operation execution, abstention, and semantic realization are separate;
- the test-manifest development exposure is visible in the table and supplement;
- protocol attacks and matched controls are retained as diagnostics;
- the final PDF, artifact manifest, and checksum identities are regenerated after
  the last table, figure, and manuscript edit.

Residual non-blocking weakness: predicted room identity, graph fidelity, and
several semantic operation rates are weak. The paper states this directly rather
than substituting contract success for semantic success.

## Domain Reviewer

Using the old CAD state as an input is a legitimate editing interface. The
manuscript describes it as an engineering prior rather than leakage and keeps the
target-derived oracle regime clearly separated from target-free predicted use.
The outside domain, wall dilation, typed-before repair, and changed-control mask
are specified with executable semantics.

## Perspective Reviewer

The practical value is conservative local editing with exact outside preservation
and calibrated abstention. The paper reports support coverage, abstention,
operation realization, raw topology, functional topology, and fidelity cost. It
does not hide unsupported requests behind a success rate.

## Devil's Advocate

Strongest counter-argument:

> Exact outside preservation follows from local composition, oracle controls
> contain target information, and predicted semantic realization remains mixed.

Current defense:

- target-free val491 structural plausibility is separately measured;
- matched Raster2Seq and editing-aware controls test freeze-only explanations;
- supported execution and unsupported abstention are independently audited;
- raw topology and semantic failures are retained;
- the paper claims a scoped reliability result, not complete SOTA.

## Required-Fix Audit

- Factual errors: corrected frozen Raster2Seq rows, table typesetting, source
  pointers, split exposure, and page-limit compliance.
- Missing explanations: added metric separation, statistical units, protocol
  boundaries, figure provenance, and held-out status.
- Optional stronger evidence: a fresh held-out run and newer checkpoint remain
  outside the scoped submission and are not represented as completed.

## Submission Boundary

The local package is ready for user review. Actual OpenReview enrollment,
submission, and any venue receipt require explicit user authorization and an
external platform action.
