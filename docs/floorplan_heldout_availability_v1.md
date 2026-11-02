# Floorplan Held-Out Availability v1

Status date: 2026-11-02

## Finding

The current CubiCasa test split does not contain a strictly fresh held-out subset.

- Total test cases: 503.
- Cases appearing in the edit-prompt report: 200.
- Cases absent from the edit-prompt report: 303.
- Full predicted-completion inference cases: 503.

The 303 cases not used for the current prompt report were still included in the
full predicted-completion inference run. They therefore cannot be described as
unopened model cases.

## Consequence

The remaining submission goal is blocked on new external evidence, not on
document polishing:

- acquire an unused CubiCasa case manifest or another compatible floorplan edit
  split;
- freeze the selected predicted model and control contract first;
- generate the new prompt manifest from case IDs only;
- run the new split once and retain every negative result.

## Current Allowed Claims

- oracle-control test503 editing evidence;
- target-free validation structural plausibility;
- operation safety and abstention on development-exposed test cases;
- preliminary semantic operation audit with disclosed failures.

No fresh held-out predicted-control claim is currently allowed.
