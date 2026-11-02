# M6 Analytical Proposal Selector Rejection

Date: 2026-10-30

Status: response repair complete; deployable selector rejected on the two-case
matched-budget pilot.

## Completed Chain

- Four missing `nangate45/bp_be_top` global-place responses completed.
- Raw arms: candidate/control x N1/N2, all `status=complete`.
- Every proposal has a finite `globalplace__route__wirelength__estimated`.
- Fixed total budget: NFE4 plus 1600 refiner steps split across proposals.

The summary builder initially rejected the repaired reports because it accepted
only `status=generated` while response-only runs emit `status=complete`. The
contract now accepts both and has a regression test.

## Pilot Result

| Evidence | Result |
|---|---:|
| Cases | 2 |
| Response seconds | 845.4 |
| Cases where analytical changes the default p2b choice | 1 |
| Cases where that change improves estimated global-place WL | 1 |
| Improvement in that case | -3.47% |

The positive case is a numerical p2b tie. Candidate `cva6` remains tied and
chooses proposal 0 in both selectors. Control `bp_be_top` and `cva6` remain tied
under both responses. There is no discriminative-p2b case where global place
changes the selected proposal.

## Decision

- Reject U2 and the deployed best-of-N selector.
- Do not integrate global-place feedback into the sampler from this pilot.
- Retain the strong cross-design prediction observed on the earlier 13 observed
  pairs as motivation, not as proposal-ranking evidence.
- Any future routing-aware route needs untouched validation proposals, a
  materially different representation, and a response cheap enough to compete
  under total-compute accounting.

## Artifacts

- `outputs/reports/chip_v46_analytical_search_report.json`
- `outputs/reports/chip_v46_analytical_search_decision_v1.json`
- `outputs/reports/chip_v46_response_repair_queue.json`
- `scripts/audit_chip_v46_search_decision.py`

