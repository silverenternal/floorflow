# Predicted-Control Topological Architecture Audit

## What Already Works

- Predicted controls pass through the staged room, wall, junction, opening, and
  furniture runtime without target leakage.
- Room-partition boundary candidates raise composed Wall F1 from `0.3396` to
  `0.3645` and Opening F1 from `0.1183` to `0.1211` on watertight-eligible
  calibration cases.
- The accepted output uses the largest legal room partition component and
  fpeval-compatible minimum room area, giving schema validity `1.0` and
  watertightness `1.0`.
- A dedicated opening decoder and room-boundary host skeleton provide a usable
  opening source.
- The candidate scorer now has wall-budget regression and a multiscale
  filtration loss adapted from TopoGED.

## What Is Still Weak

1. The eight-epoch scorer without filtration loss peaks at epoch 0 and remains
   below the deterministic candidate control. It is rejected.
2. The filtration descriptor is computed from candidate scores after scoring;
   it is not predicted by the network before reconstruction.
3. Candidate scores are produced independently by an MLP. There is no
   candidate-to-candidate graph attention over shared junctions.
4. There is one global wall budget. Exterior walls and interior partition walls
   have no separate budgets, unlike TopoGED's phased edge reconstruction.
5. Candidate geometric residuals are not conditioned on the predicted
   topological descriptor.
6. Raw candidate labels are stable, but room-to-wall source incidence is not a
   graph feature available to the scorer.
7. Descriptor quality, budget error, phase-count error, and junction-degree
   error are not yet first-class reported metrics.

## Architectural Upgrade

The next probe replaces the standalone candidate MLP with a descriptor-first
graph decoder:

```text
staged predicted controls
  -> graph-token topological descriptor predictor
       (8 filtration counts + exterior/interior wall budgets)
  -> candidate graph transformer
       (shared-junction attention + footprint-phase embedding)
  -> descriptor-conditioned candidate scores
       (exterior budget, interior budget, global total)
  -> bounded endpoint snapping
  -> phased top-K reconstruction
  -> host-wall openings and fpeval validation
```

Required network changes:

- Predict the multiscale descriptor from ordered CAD stage tokens.
- Condition candidate representations on the predicted descriptor through
  FiLM/residual modulation.
- Run masked multi-head attention between wall candidates connected at shared
  endpoints.
- Predict exterior and interior wall budgets separately from graph context.
- Reconstruct each phase with its own top-K budget.
- Supervise predicted descriptor, constructed descriptor, total budget, phase
  budgets, candidate existence, and bounded endpoints.

## Fast Iteration Budget

The user-approved fast screen uses:

- 64 train cases and 32 calibration cases;
- eight epochs;
- batch size 4;
- `cuda:0`;
- one shared frozen layer checkpoint and identical opening/room composition.

This is a mechanism screen only. Any positive mechanism must later be confirmed
on at least train256/cal64 before promotion.

## Control

The matched control is the deterministic candidate evidence threshold `0.2`
inside the same small split. All reported deltas use the same validation cases.

## Immediate Gates

- predicted descriptor and budget losses decrease over at least the final three
  checkpoints;
- candidate graph residual does not collapse selected-wall count to zero;
- composed watertight-eligible wall F1 exceeds the small-split deterministic
  control;
- opening and room metrics do not regress;
- all-case schema validity remains `1.0`.

## Fast-Screen Audit

The first 64/32 graph-decoder screen was decomposed using the same checkpoint
and validation cases:

| Decode arm | Watertight-eligible Wall F1 |
| --- | ---: |
| Deterministic evidence control | 0.3539 |
| Raw candidates + learned score | **0.3638** |
| Refined endpoints + learned score | 0.3326 |
| Raw candidates + predicted budget top-K | 0.3309 |
| Refined endpoints + predicted budget top-K | 0.3045 |

This isolates a real result: descriptor-conditioned candidate scoring improves
the deterministic candidate selector, but endpoint residual and budget-as-final-
selector both hurt. The next screen therefore keeps descriptor and budget
prediction as score conditions and auxiliary topology supervision, while using
learned score for reconstruction and disabling residual decoding.

The subsequent score-only fast screen reached `0.3629` at epoch 2 against the
same `0.3539` control, but later epochs overfit/collapsed. The mechanism is
positive but not stable enough for promotion; no full-split run is authorized.

After anchoring score to the candidate-evidence logit, the 64/32 run starts
directly at `0.3587` eligible Wall F1 and remains above the `0.3539` control at
epoch 0, but the learned correction is still too weak to produce a durable
trend. Compact residual and lower-learning-rate runs confirm that the remaining
problem is training signal/data scale for the residual itself.

Engineering fixes completed in this pass:

- empty type-match and empty-candidate batches no longer produce NaN metrics;
- new junction-stage wiring is opt-in, so frozen checkpoints keep their original
  contract;
- the scorer starts as an exact evidence-logit prior;
- residual magnitude uses a bounded evidence scale, compact topology features,
  and L2 regularization;
- candidate-plan evaluation moved out of the training script;
- program suppression reuses the existing `apply_program_constraint`;
- all changed code passes scoped Ruff, `878` tests, and the 600-line limit.

## Stability Fix

The candidate wall evidence is now a fixed logit prior:

```text
score_logit = descriptor_conditioned_residual + evidence_scale * logit(evidence)
```

At initialization the residual is zero and `evidence_scale=1`, so the network is
exactly the deterministic candidate control. Descriptor, graph attention, type,
and budget heads can only learn an incremental correction from a known safe
starting point. This removes the observed epoch-0 ranking drift.

