# Fixed-Outline L0 Conditioning v1

Status date: 2026-10-30.

## Finding

The current model has no explicit external-outline input. It predicts an L0
bounding-box state from block areas, constraints, netlist and pins, but a
bookshelf case can request a specific `(W, H)` outline that is not represented
in those inputs.

Two GSRC/MCNC smoke cases at NFE4/refiner400 produced zero feasible layouts with
4 draws and 16 draws. Increasing the existing refiner outline weight from `1` to
`10` and `100` also produced zero feasible layouts. This rejects a simple
draw-budget or outline-loss-weight explanation.

## Smallest Architecture-Aligned Change

Condition the existing L0 state; do not add a parallel encoder or a new
trajectory.

1. Add an optional normalized outline vector
   `[cx/s, cy/s, W/s, H/s]`, where `s=sqrt(sum(block area))`.
2. Add it to `batch` as `outline_state`.
3. In `HierarchicalTokenFlowModel`, create an opt-in zero-initialized
   `outline_to_hidden` projection only when the config enables it.
4. Add that embedding to the existing hierarchy summary and typed global token
   before `global_head` and `hierarchy_to_latent`.
5. Keep the option disabled by default so all current checkpoints load without
   a state-dict migration.
6. Add an outline-consistency objective on the decoded global bbox and the
   existing bbox supervision; do not add a separate network head.

## Training Data Contract

FloorSet does not carry a user-specified outline, so the training transform must
create a valid conditioning target:

- derive the reference center and aspect from the GT placement bbox;
- sample deterministic outline margins from a frozen set such as
  `{0%, 5%, 10%, 20%}`;
- keep the outline centered on the GT envelope;
- record the sampled outline state in the packed batch;
- keep the same outline for every candidate/control comparison.

For bookshelf evaluation, use the protocol outline from
`with_fixed_outline(gamma=0.10, aspect=1.0, map_pins=False)`.

## Promotion Gate

1. Unit tests: missing outline reproduces the base model exactly when the option
   is disabled; zero-initialized enabled path is base-equivalent at step zero.
2. Frozen FloorSet subset: no regression in legalized cost or V_rel at
   2k/refiner400.
3. Frozen FloorSet confirmation: improvement must remain at 5k/refiner1600.
4. Bookshelf screen: increase outline-feasible fraction from zero on the two
   smoke cases before expanding to all 22 cases.
5. Full bookshelf: HARD and SOFT reported separately; raw HPWL is reported only
   among feasible layouts.

This task is blocked behind the current M6 decision and A17 resource gate. It
does not replace the main per-block trajectory claim unless it passes the
matched full-data candidate gate.

## Implementation Status

The opt-in path is implemented but untrained:

- `outline_condition_from_boxes` and `outline_condition_from_hw` normalize the
  condition by `sqrt(sum(block area))`.
- `HierarchicalTokenFlowModel(outline_conditioning=True)` adds one
  zero-initialized `4 -> hidden` projection at the existing L0 summary/global
  path. Its initialization is RNG-isolated so candidate/control matched runs
  retain the same downstream random sequence.
- `outline_conditioning=False` adds no parameters and preserves old
  checkpoints.
- Training derives centered outline states from GT envelopes with a configurable
  maximum margin.
- Bookshelf evaluation supplies the explicit outlined `(W, H)`.

Unit tests cover helper normalization, missing-state rejection, disabled-path
parameter absence and zero-initialized enabled-path equivalence. Training and
the subset promotion screen remain blocked behind M6/A17.

A one-step CUDA smoke completed the generated-outline loss, backward pass, EMA
update, dev evaluation and checkpoint save. The saved checkpoint reloads with
`outline_conditioning=True` and a `192 x 4` projection. This verifies the
integration path only; no quality claim is attached to the one-step model.

## U3 fixed-outline fallback

The opt-in L0 condition alone did not remove the zero-feasibility result.
Increasing the refiner outline weight to 10 or 100 also failed. A deterministic
fallback was therefore added after exact-area decode and the existing
scale-pack route:

1. restore fixed shapes through scale-pack;
2. try the existing deterministic shelf pack;
3. if the shelf pack cannot fit, solve a fixed-rectangle MILP with optional
   hard-block rotation and an objective favoring the incoming positions;
4. accept the result only after exact outline and pairwise non-overlap checks.

The two-case structural smoke used 4 draws per arm:

| Arm | Outline-feasible cases | n10 feasible draws | APTE feasible draws |
|---|---:|---:|---:|
| candidate | 2/2 | 3/4 | 2/4 |
| control | 1/2 | 2/4 | 0/4 |

This is feasibility evidence only. HPWL is not compared on infeasible layouts,
but the frozen 22-case GSRC/MCNC cohort was then opened at 4 draws per arm:

| Arm | Feasible cases | Feasible fraction |
|---|---:|---:|
| candidate | 10/22 | 0.4545 |
| control | 9/22 | 0.4091 |

Only seven cases are feasible on both arms. Their feasible-pair mean HPWL
relative is `+1.87%` for the candidate, so the result is not a performance win.
The fallback improves outline-feasibility coverage but has substantial
case-to-case heterogeneity. The evaluator now reports `mean_hpwl_all`,
`mean_hpwl_feasible`, and a paired HPWL delta only for double-feasible cases.
The next gate is 16 draws on the feasible/ambiguous subset; this remains
external structural evidence, not ORFS routed-PPA evidence.

### 16-draw stability result

Running all 22 cases at 16 draws per arm yields:

| Measurement | Candidate | Control |
|---|---:|---:|
| Feasible cases | 13/22 | 9/22 |
| Feasible fraction | 0.5909 | 0.4091 |

The discordant feasibility counts give exact McNemar `p = 0.21875`. Eight
cases are feasible on both arms; their mean HPWL delta is `-3120.38`
(`-0.94%` relative), with paired bootstrap CI `[-15557.04, +9266.69]`.

The U3 fallback therefore has a positive but statistically uncertain
feasibility-coverage signal. It is retained as a structural diagnostic, not a
performance or routed-PPA claim.

The HARD/SOFT split is equally important:

| Variant | Candidate feasible | Control feasible | Paired feasible | Paired HPWL |
|---|---:|---:|---:|---:|
| HARD | 6/11 | 6/11 | 6 | -0.37% |
| SOFT | 7/11 | 3/11 | 2 | -1.08% |

The feasibility signal is SOFT-concentrated, while HARD feasibility is tied.
Neither variant's paired HPWL interval excludes zero. Do not generalize the
fallback as a HARD-outline improvement.

A targeted 10-second MILP probe on ten persistent failures produced zero fully
feasible candidate/control layouts. Several soft cases reached
`fits_outline=true` but failed another feasibility check, so the remaining
failures are not explained by the default 2-second MILP time limit.

### Training-seed probe

The same two-case, four-draw probe was repeated with matched seed20 and seed21
candidate/Trinity checkpoints:

| Training seed | Candidate feasible | Control feasible |
|---:|---:|---:|
| 19 | 2/2 | 1/2 |
| 20 | 1/2 | 0/2 |
| 21 | 2/2 | 1/2 |

The direction is consistent, but APTE remains unstable and the two-case HPWL
signs are mixed. This is a seed-stability diagnostic, not a performance claim.

Pooling the six matched seed-case comparisons gives candidate feasibility
`5/6` versus control `2/6`; exact McNemar `p=0.25`. Only the two double-feasible
n10 cases can be compared on HPWL, with mean `-3.43%`. Neither statistic supports
a performance claim.

### Draw-level failure attribution

The 16-draw cohort has 352 draws per arm:

| Variant | Candidate fully feasible | Control fully feasible | Remaining mode |
|---|---:|---:|---|
| HARD | 79/176 | 79/176 | 97 outline-only misses per arm |
| SOFT | 55/176 | 40/176 | outline/overlap mixed |

No draw has a shape/area violation. With fallback enabled, HARD candidate and
control have exactly the same feasibility-mode counts. Thus the
candidate-versus-control advantage is SOFT-concentrated and cannot support a
HARD-outline model claim. The fallback itself still raises HARD coverage versus
the no-fallback control described below.

### Fallback on/off effect

The same 22-case, 4-draw cohort was run with the fallback disabled:

| Arm | With fallback | Without fallback | Gains / losses | Exact McNemar |
|---|---:|---:|---:|---:|
| candidate | 10/22 | 2/22 | 9 / 1 | p=0.0215 |
| control | 9/22 | 1/22 | 8 / 0 | p=0.0078 |

HARD feasibility rises from `0/11` without fallback to `5/11` (candidate) and
`6/11` (control). The candidate's only loss is `mcnc/soft/ami33`; all other
transitions gain feasibility. This is a mechanism-level coverage result. HPWL
has only one paired-feasible case per arm and remains diagnostic.

At the primary 16-draw budget, the effect strengthens:

| Arm | With fallback | Without fallback | Gains / losses | Exact McNemar |
|---|---:|---:|---:|---:|
| candidate | 13/22 | 2/22 | 11 / 0 | p=0.00098 |
| control | 9/22 | 1/22 | 8 / 0 | p=0.0078 |

For HARD cases, both arms rise from `0/11` to `6/11` feasibility. The fallback
therefore has a clear fixed-outline coverage mechanism effect. It still has only
two candidate and one control paired-feasible HPWL cases, so no performance
claim is allowed.

Matched seed20/21 two-case on/off probes also show no losses:

- seed20: candidate `0→1`, control `0→0`;
- seed21: candidate `0→2`, control `0→1`.

The direction is consistent with seed19, but each seed probe contains only two
cases and is not independently significant.
