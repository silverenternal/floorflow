# S3 Budget-Matched Candidate Subset Audit

Date: 2026-11-01

## Protocol

- candidates: M48 seed19, M48 seed20, retained M48 seed21, M61B-P32, M65;
- every subset keeps M48 seed19 as the reference and uses exactly four views;
- selector training cases: 408;
- validation cases: 92;
- anchor: Trinity d192 NFE16/ref1600 on the same 92 cases;
- selector hyperparameters and utility weights stayed frozen;
- fresh500 was not read.

## Result

| Candidate views (excluding seed19) | Joint | Strict | Safe | Oracle (cost/V_rel/group/rho) |
| --- | ---: | ---: | ---: | ---: |
| seed21, M61B, M65 | 51 | 26 | 8 | 24/92 |
| seed20, seed21, M61B | 49 | 20 | 8 | 26/92 |
| seed20, seed21, M65 | 49 | 22 | 7 | 23/92 |
| seed20, M61B, M65 | 45 | 22 | 7 | 23/92 |

All numbers are `wins/92`.

The validation-only winner replaces seed20 with seed21 and keeps M61B/M65:

- cost `-1.32%`;
- V_rel `-5.68%`;
- grouping `+1.59%`;
- rho MAE `+4.93%`;
- boundary `-17.05%`;
- joint `51/92`;
- strict structural `26/92`;
- constraint-safe `8/92`.

The winning subset still fails every coverage gate. The strongest possible
candidate-pool oracle reaches only `24/92` for simultaneous
cost/V_rel/grouping/rho non-regression, below the `35%` target.

## Decision

Do not promote any budget-matched 4-view subset. Changing or reweighting this
portfolio cannot reach the case-coverage gate because the candidate pool itself
does not contain enough anchor-relative safe outcomes.

Seed21 may remain as a diagnostic fifth view only under an explicit NFE20/ref2000
budget. It must not replace a view to claim a matched NFE16 win without a fresh
confirmation. Further progress requires a new legalizer-compatible candidate
generator for grouping/rho, not another selector sweep.

## Artifacts

- `scripts/audit_chip_candidate_subsets.py`
- `src/floorflow/chip_flow/candidate_subset_audit.py`
- `tests/test_chip_candidate_subset_audit.py`
- `outputs/reports/chip_s3_candidate_subsets_val92_v1.json`
- `outputs/reports/chip_trinity_validation92_nfe16_ref1600_v1_cases.json`

