# Chip Architecture Salvage Review v2

Status: CPU evidence audit only. No GPU or ORFS experiment was run.

The matrix separates failed output contracts from reusable operators.
A reusable component is not evidence that its old mechanism worked.

| Family | Decision | Reuse target |
|---|---|---|
| M36-M39 hidden hierarchy bridge and bbox consensus | closed | M50 same-call state commitment, not the rejected residual. |
| M40 self-conditioned flow | closed | Runtime diagnostics only; do not reopen keep-probability sweeps. |
| M45 projector-displacement output head | closed | M53 endpoint compatibility, without projected-output substitution. |
| M46 all-layer state-conditioned edge bias | closed | M52 current-state dynamic graph input with relation-specific fusion. |
| M10 final-only projection-consistent flow | closed | M53 endpoint validation only. |
| U1 additive dynamic graph relay | closed | M52 dynamic graph-input conditioning. |
| Spatial occupancy, port consensus and ordered group staging | reusable_component | M52/M55 typed graph and routeability fields only. |

## Rules

- Do not rerun a closed gate with width, gate, mask, or scale sweeps.
- Preserve legalizer rung/solve counts in every future mechanism probe.
- Reopen only with a new state contract and a matched control.

Machine-readable matrix: `pass_cpu_audit`.
