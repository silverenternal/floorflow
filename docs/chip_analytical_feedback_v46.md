# Chip Analytical Feedback v46

## Decision

The next chip mechanism is an explicitly budgeted analytical-placement response,
not another in-model scalar routing proxy or loss-weight sweep.

The existing model exposes macro geometry, typed b2b/p2b connectivity, clusters,
MIB groups, boundary constraints and ports. It does not expose the standard-cell
distribution produced when a legalized macro placement is handed to a
mixed-size placer.

## Observed development evidence

`scripts/audit_chip_downstream_signal_v46.py` joins certified v45 endpoint pairs
to OpenROAD Metrics_2.1.2 stage metrics.

- 13 completed candidate/control pairs, 13 designs, 7 RTL families.
- `globalplace__route__wirelength__estimated` versus final routed wirelength:
  design-level Spearman `0.9774` (`p = 9.03e-9`).
- Leave-one-design-out minimum: `0.9710`; every omitted-design sign agrees.
- Leave-one-family-out minimum absolute Spearman: `0.9630`; every sign agrees.
- Binary candidate/control selection has correct sign on `13/13` observed pairs,
  mean routed-WL change `-0.72%`, worst selected change `0.00%`.
- Median global-place response time: `210.43 s`; p90 `571.20 s`; maximum
  `718.21 s`.
- The old p2b reference on the legacy 18-pair diagnostic has Spearman `-0.135`
  (`p = 0.593`) with routed WL.
- Thresholding p2b for candidate/control selection is correct on only `8/18`
  legacy pairs and yields `+0.91%` routed-WL change; the analytical response is
  therefore not interchangeable with the current macro-only surrogate.

A retrospective five-design portfolio over the three already trained candidate
model seeds further improves the mean routed-WL change from `+0.75%` at fixed
seed 19 to `-1.98%` against each selected model seed's matched control. This is
only headroom evidence: every proposal still used the full refiner budget, and it
is not a same-checkpoint noise-search result.

The predeclared three-case pilot now has generated proposal sets. N=1 uses
NFE4/refiner1600; N=2 splits the same totals into NFE2/refiner800 per proposal.
Both sets were regenerated with `shelf_pack_forced` after route mismatches were
observed before any response outcome. Under that common legalizer, p2b HPWL is
identical for cva6 and bp_be_top but retains a 24.27 um range on tinyRocket,
where all arms can use scale-pack without destroying proposal diversity. The
machine-checked preflight artifact is
`outputs/reports/chip_v46_matched_flow_preflight.json`.

The frozen preflight was refreshed with position-aware shelf packing after a
size-only shelf collapsed equal-sized proposals. Every case now retains five
unique TCL hashes across its six candidate/control proposal files. On cva6 and
bp_be_top, p2b remains tied even though the TCL placements differ, which makes
those cases the direct test of whether analytical feedback carries information
that the macro-only surrogate lacks.

## Protocol

1. Generate a predeclared number of legalized proposals.
   If their legalizer routes differ within a case, regenerate every proposal in
   that case through the same deterministic fallback before scoring.
2. Split the total refiner budget across proposals.
3. Run through ORFS target `3_3_place_gp`, which builds prerequisites but stops before resize/detail placement/routing.
4. Rank by `globalplace__route__wirelength__estimated`; lower is better.
5. Count model sampling, legalization and global placement in the total budget.
6. Route only the selected proposal on the screening subset.
7. Keep detailed route, DRC, congestion, timing, power and area as final
   endpoints. They must not enter proposal selection.

`scripts/run_chip_analytical_response.py` implements step 3 with the pinned ORFS
image and source revision. The response wrapper excludes detailed-route and
post-route metrics by construction. `scripts/evaluate_chip_analytical_search.py`
drives proposal generation and response scoring under the frozen budgets in
`configs/chip_v46_analytical_search.json`; `--generate-only` and
`--response-only` split GPU proposal generation from CPU/OpenROAD scoring so the
two resources do not have to wait on each other.

`scripts/build_chip_v46_analytical_search_report.py` combines all four arms into
one report. It checks the fixed NFE/refiner contract, detects exact p2b ties,
records whether analytical scoring resolves them, compares candidate against
control, and compares N2 against N1 under the same total generation budget. A
partially completed response remains explicitly pending rather than being
selected from whichever proposals happen to be available.

`scripts/run_chip_v46_response_queue.py` monitors the existing OpenROAD PID and
runs the frozen response-only queue after that run exits. The queue never
modifies or replaces A17.

The response wrapper reuses invariant synthesis and initial-floorplan artifacts
from the existing `macro_count` variant before running the new `3_3_place_gp`
variant. On tinyRocket this reduced a response from about `100s` including
synthesis to `36–38s`. The two scale-pack proposals scored `401544` and `395731`;
both p2b and the analytical response selected proposal1, so tinyRocket is an
agreement case rather than evidence of a unique analytical gain.

## Boundaries

- The v45/v44 observations are development evidence, not fresh confirmation.
- Global-place setup WS/TNS deltas do not transfer to final timing
  (Spearman `0.017` and `-0.149` over 13 pairs), so timing remains an independent
  post-route guard rather than part of the selection score.
- The observed binary selection cannot support a deployment claim by itself.
- The retrospective seed portfolio is not equal-budget.
- Global placement is an external physical response model, not a zero-cost
  surrogate; its latency must be reported.
- A new mechanism requires a matched-flow-budget proposal screen, explicit
  proposal-dependent EDA cost, and then a non-overlapping validation set before
  model integration or full-data training.

The untouched external-set audit is separate from the three-case mechanism
pilot. The v45 set is already observed and cannot be reused as confirmation. The
static supply audit found only three plausible unused macro-capable configs on
installed open platforms; the eight GF12 configs depend on an NDA-restricted
platform that is absent from the pinned checkout. The 15-design fresh cohort is
therefore supply-blocked until macro counts are measured and additional
open-license hierarchical cases are added.

Pair integrity is also a hard gate. `scripts/audit_chip_external_pair_integrity.py`
shows that all 18 v44 design-seed comparisons used distinct legalized TCL files,
so the v44 negative result remains valid. In v45, seven candidate/control cases
collapsed to identical TCL hashes after legalization; only eight of the 13
completed cases are usable model comparisons. These original outcomes stay
archived and a corrected rerun can only be a separate sensitivity cohort.

The corrected sensitivity cohort applies a rigid core translation when
`scale_pack` already returns a feasible zero-overlap layout but the whole
packing rectangle lies slightly outside the core. All seven formerly collapsed
pairs now produce distinct TCL hashes with a common legalizer route. This fixes
an upstream proposal-to-layout collapse and is now a precondition for any new
external PPA run.

## Open-source integration decision

DREAMPlace (`6627f3327e6cc17db7782c0b90073a498531ca3c`, BSD-3-Clause) provides a
GPU/CPU analytical placement stack and is the strongest candidate if response
latency becomes the bottleneck. It is not vendored or substituted now because
the current evidence was calibrated against the pinned OpenROAD global placer.
Changing the physical response before proposal-level validation would create an
unmeasured protocol shift.

MacroDiff-plus (`c28ef15bc9ebba6c80eedaf482beb322a7ede4b1`, MIT) is a useful
guidance reference, but its MMS/ISPD model and metrics are not interchangeable
with FloorSet/ORFS. ChipDiffusion (`6973e90ee1afc345b56da486bbf7dc9772ada1a4`)
has no repository license, so its code is not a legal reuse source; its
published guidance idea may only inform a clean implementation.
