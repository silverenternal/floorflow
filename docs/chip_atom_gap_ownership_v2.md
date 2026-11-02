# Chip Atom-Gap Ownership v2

Date: 2026-11-02

## Executive Finding

The current selector already predicts multiple legalized metrics, but it aggregates
them with fixed scalar utility weights. It does not train independent atom experts
or learn a case-conditioned typed combination.

The existing typed graph stack already provides the reusable pieces:

- `TypedChannelState` separates global, structural, constraint, block and port
  tokens.
- `TypedCrossAttention` supports sparse allowed edges, type biases,
  relation-specific gates and zero-gated residual fusion.
- `training_scopes.py` already isolates channel and atom families:
  `block`, `local`, `global`, `structural`, `constraint`, `port`, `semantic`,
  `expert`, `bridge`, `area`, `b2b`, `port_route`, `boundary`, `tail`,
  `fusion` and `downstream`.
- `typed_channel_cache.py` can cache frozen pre-fusion states, so atom heads and
  fusion experiments do not need to rerun the encoder.

The missing layer is not another backbone. It is an atom-evidence interface:
independent metric heads over existing candidate/channel features, followed by a
typed cross-attention gate learned on out-of-fold atom predictions.

## Current Evidence

All values below are legalized and use the declared candidate budget.

| Atom | Current deployed evidence vs anchor | Existing owner | Main failure |
|---|---|---|---|
| Grouping | `+16.22%` mean, CI crosses zero in the S3 confirmation; candidate pool is the limit | A07/A08/A11/A15/A16/A19/A20 | selector does not consistently recover grouping oracles |
| Area / outline / rho | area `-15.01%`, CI crosses zero; rho `+8.38%`, CI excludes zero | A05/A18/A24/A27/A33/A34 | area and rho cannot both be optimized by one scalar utility |
| b2b connectivity | `-21.16%` mean, CI crosses zero | A01/A02/A03/A04/A08/A11/A17/A24 | 182 improved while 318 worsened versus anchor |
| Boundary | `-26.74%`, CI excludes zero | A05/A09/A18/A26/A27 | strong mean gain but still candidate-dependent |
| p2b / port | `-79.78%`, CI excludes zero | A10/A14/A21/A22/A25/A28/A36 | structural gain does not consistently transfer to routed WL |
| Legality / cost tail | rung and solver calls improve; hard feasibility is the guard | A23/A29/A32/A33/A34/A35 | tails and legalizer escalation dominate cost |
| MIB | tied at zero | A08/A09/A24/A31 | no discrimination; diagnostic only |

The strongest current Trinity component gaps remain:

- b2b HPWL gap: `+41.50%` relative in the older mechanism row;
- area gap: `+24.83%`;
- grouping: `+33.19%`;
- rho MAE: `+10.92%`.

The deployed selector closes pooled cost and V_rel but still loses grouping and
rho. This makes grouping and rho the first atom targets.

## Hard Coverage Bound

On the frozen 500-case confirmation pool:

- selected joint wins: `254/500`;
- selected constraint-safe joint wins: `40/500`;
- candidate-pool four-metric oracle: `132/500`;
- candidate-pool all-metric oracle: `25/500`.

The CSJ target is `175/500`. Even a perfect selector over the current pool cannot
reach it. Therefore:

1. Atom heads and typed aggregation can recover useful selection regret, but cannot
   alone produce the required coverage.
2. New atom-specific candidates are mandatory, especially candidates that jointly
   improve grouping/rho with area/b2b.
3. Any claimed coverage gain must report the pool oracle before and after adding
   candidates.

## Six Atom Contracts

### 1. Grouping

- Deployment labels: `legalized_grouping`, grouped `V_rel`.
- Reused features: structural tokens, membership, cluster incidence, block tokens.
- Existing heads: structural relation head, hierarchy coupler.
- Screen proxies: grouping proxy, cluster relation MAE, contact-edge consistency.
- Required new candidate: preserve grouping while changing area/b2b.

### 2. Area / Outline / Shape

- Deployment labels: `legalized_area_gap`, `legalized_rho_mae`.
- Reused channels: global spec, complexity, block/factorized head, boundary bridge.
- Existing scopes: `area`, `boundary`, `downstream`.
- Screen proxies: exact-area residual, outline compactness/aspect, rho MAE.
- Required new candidate: area-safe rho alternatives, not rho reranking.

### 3. b2b Connectivity

- Deployment labels: `legalized_b2b_hpwl_gap`, cost p90/p95.
- Reused channels: local geometry, structural, degree expert, hyperedge conditioner.
- Existing scope: `b2b`.
- Screen proxies: raw/legalized b2b sign agreement, degree reconstruction,
  high-degree stratum.
- Required new candidate: b2b candidate that does not trade away grouping.

### 4. Boundary

- Deployment labels: `legalized_boundary`.
- Reused channels: constraint tokens, coordinate gate, boundary anchor,
  global bbox consensus.
- Existing scope: `boundary`.
- Screen proxies: boundary contact error, edge alignment, boundary displacement.
- Boundary has a strong mean gain; priority is preserving it during DBU and
  aggregation, not adding another boundary-only loss.

### 5. p2b / Port

- Deployment labels: `legalized_p2b_hpwl_gap`.
- Reused channels: port input/type, port consensus, dual port head,
  spatial occupancy.
- Existing scope: `port_route`.
- External truth: routed WL, congestion and DRC.
- p2b is a diagnostic until routed sign agreement is non-negative. It must not be
  the only reason a candidate is promoted.

### 6. Legality / Cost Tail

- Deployment labels: hard-feasible rate, cost p90/p95, legalizer rung/solves,
  displacement p95.
- Reused channels: legalizer response, complexity slot, projector, scale_pack,
  DBU projection.
- Existing scope: `tail`.
- This atom owns fail-closed guards and should veto, not be averaged into utility.

## Reusable Model Organization

### Atom head bank

Add one small head per atom over a shared frozen feature encoder:

- input: existing 14 target-free candidate features plus masked pooled typed
  context tokens;
- output: normalized atom metric and log variance;
- loss: masked Smooth-L1 plus heteroscedastic or rank loss;
- size target: two 64-wide layers per atom, under 0.5M new parameters total;
- training: atom heads can run independently with `apply_training_scope`.

This is not a new backbone and does not require changing the flow endpoint.

### Typed atom aggregator

Use the existing `TypedCrossAttention` rather than a dense selector:

- query: six atom evidence tokens;
- memory: global, structural, constraint, port and block tokens;
- allowed mask:
  - grouping <- structural + block + constraint;
  - area/rho <- global + block + constraint;
  - b2b <- structural + block + global;
  - boundary <- constraint + global + block;
  - p2b <- port + structural + block;
  - tail <- global + constraint + block;
- second sparse pass between atoms:
  - grouping <-> b2b;
  - grouping <-> area;
  - boundary <-> area;
  - boundary <-> p2b.

The gate predicts candidate utility and per-atom reliability. Legality remains a
hard veto outside the average.

### Leakage-safe training

1. Produce case-disjoint out-of-fold atom predictions for the 408 training cases.
2. Train the typed aggregator only on those OOF predictions.
3. Keep the 92-case development split untouched until the aggregator and all atom
   heads are frozen.
4. Use fresh 500 only for confirmation.

## Three Parallel P0 Tracks

### A. Atom Heads

Budget: CPU, 408 cases x 4 candidates, 5-fold case split.

Deliverables:

- six independent metric heads;
- fold-wise calibration, Spearman, pairwise sign agreement, top-3 recall and regret;
- fixed-selector comparison.

Gate: every promoted head passes proxy-real screen at >=15 cases and shows no
high-complexity sign reversal. This track can start immediately.

### B. Typed Aggregator

Budget: CPU cache + optional GPU0, at most 2k updates after fusion cache exists.

Deliverables:

- pre-fusion typed cache for the frozen 500-case cohort;
- typed/sparse attention aggregator;
- dense, no-structural and no-constraint ablations.

Gate: positive gains over fixed utility on grouping/rho JSR without losing
constraint-safe wins. Because the current pool oracle is only 25/500 all-metric,
this track is a regret-recovery layer, not the coverage solution.

### C. Atom-Specific Generators

Budget: GPU0 small screens on 300-1000 frozen train cases, 300-1000 updates per
channel. Start with grouping and rho/area because they are the dominant remaining
anchor gaps.

Deliverables:

- one candidate generator per atom family, using existing channel scopes;
- pool-oracle measurement after each generator;
- no full-data run until the oracle increases on the 500-case development cohort.

Gate: add candidates that increase the four-metric or all-metric pool oracle.
Generator acceptance is by legalized metrics only.

## Execution Order

Parallel now:

1. Atom-head CPU screen on existing candidate features.
2. Typed-channel cache for the frozen cohort.
3. Grouping and rho/area generator design, default-off and scoped.

After the first pass:

1. Train the typed aggregator on OOF atom predictions.
2. Merge only oracle-increasing candidates.
3. Run a 500-case development screen, then a new fresh confirmation.

Do not start another dense all-metric selector. The current failure is not missing
capacity; it is missing independent atom evidence, learned sparse aggregation and
new legalizer-compatible candidates.
