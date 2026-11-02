# Chip Architecture Upgrade Research v1

Date: 2026-10-30

Status: planning evidence only. No experiment in this document is a new model
result for FloorSet or ORFS.

Machine-readable companion:
`configs/chip_upgrade_candidates_v1.json`.

## 1. Scope and decision rule

This review starts from the current blockers rather than from a list of recent
papers:

1. The candidate loses both legalized cost and V_rel to the equal-weight
   group=2 Trinity d192 anchor.
2. External routed PPA is negative on the valid observed v45 pairs.
3. Macro-only p2b/b2b proxies do not transfer reliably to routed wirelength.
4. Fixed-outline generation currently has zero feasible samples on the frozen
   bookshelf screens.
5. A17 detailed routing is still occupying the CPU/OpenROAD lane, so new GPU
   training and new OpenROAD jobs must wait.

A paper is admitted to the implementation queue only if it maps to an existing
token, state, operator, sampler stage, objective, projector or external
evaluator. A generic extra layer, dense attention block, PDN/CTS/router head, or
another scalar loss is not an admissible reading of a paper.

## 2. Current code-path audit

### 2.0 Projection-consistent flow gap

The DAC'26 primary source [FlowPlace, arXiv:2604.23658](https://arxiv.org/abs/2604.23658)
and its [official implementation](https://github.com/lamda-bbo/flowplace) expose
a mechanism that maps directly to our existing flow rather than requiring a new
backbone. At each sampling step, the released recipe predicts an endpoint,
projects it onto the hard-constraint set, computes a corrected velocity toward
that feasible endpoint, and reinterpolates the trajectory.

Our current `integrate_hierarchy_step` applies the hard projector after each Euler
advance and then continues from that projected state. It preserves exact area
and protected blocks, but the next interpolation is not anchored to the
projected final endpoint. The opt-in M10 path adds this missing operation using
our existing `HierarchicalTokenFlowModel`, source state and
`project_hard_constraints`; it adds no model parameters.

The official FlowPlace repository has no license file as of 2026-10-30. It is a
paper-level mechanism reference only. M10 was implemented as a clean-room
extension of our existing integrator and tested against default post-step and
final-only projection arms. The final-only quick gain at refiner400 reversed at
refiner1600 in both primary metrics, so M10 is now a retained confirmation
negative and does not advance to full data.

### 2.1 Netlist messages are static across the flow trajectory

`HierarchicalTokenFlowModel.prepare_context` calls `TypedGraphEncoder` once.
`HierarchicalTokenFlowModel.forward` and `integrate_hierarchy_step` then reuse
that context for every ODE step. The backbone adjacency is the static b2b
adjacency supplied by `graph_conditioned_features`.

Consequences:

- the current block token changes during sampling;
- its current center, width, height and net-span relation do not modify the
  netlist message path;
- the model has no per-step graph `G_t`;
- a static source-target edge gate cannot represent this missing signal, which
  is consistent with the A5 quick/confirmation/full-data reversal.

This is an **absent dynamic path**, not evidence that graph conditioning itself
is weak.

### 2.2 External analytical response is post-generation only

`analytical_feedback.summarize_analytical_response` and the M6 search driver rank
already generated proposals. The response is not exposed to the flow or
refiner. This is the correct conservative starting point because the earlier
A15/A16 in-model scalar routing proxies failed.

The current response path is therefore:

```text
generated and legalized proposals -> global-place response -> selection
```

It is not:

```text
response -> model conditioning -> generation
```

Any model integration must first prove incremental ranking value and then use a
strictly bounded feedback path.

### 2.3 Fixed outline enters L0 but not the complete feasibility interface

`outline_condition_from_boxes`, `outline_condition_from_hw`, and the zero-init
`outline_to_hidden` projection provide an opt-in L0 signal. The external failed
draws show that this scalar L0 context alone is insufficient. The projector
still treats local block constraints primarily, without an explicit
outline-manifold fit stage.

### 2.4 Search restarts proposals but cannot mutate a promising layout

The current equal-budget and analytical search paths generate complete
proposals. There is no state-preserving partial reflow, critical-block selection
or bounded local reconstruction. This differs from partial rip-up/re-place and
diffusion mutation, where a parent structure is retained while selected
variables are re-optimized.

## 3. Verified primary sources

All arXiv identifiers were checked through the arXiv Atom API on 2026-10-30.
Reported numerical gains below belong to the cited papers, not to this project.

| Source | Relevant mechanism | Why it maps here |
|---|---|---|
| [DAS-MP, arXiv:2505.16445](https://arxiv.org/abs/2505.16445) | Macro-cell-cluster dataflow, macro-area congestion tuning | The current macro-only p2b graph omits standard-cell clusters and their spatial demand |
| [Chip Placement with Diffusion, arXiv:2407.12282](https://arxiv.org/abs/2407.12282) | Guided sampling for zero-shot placement | Supports inference-time physical feedback, but only after reward fidelity is validated |
| [Noise Trajectory Search, arXiv:2506.03164](https://arxiv.org/abs/2506.03164) | Equal-budget step-level noise search | Supports equal-compute trajectory search; best-of-N alone already failed here |
| [MacroDiff+, arXiv:2605.16451](https://arxiv.org/abs/2605.16451) | Dynamic heterogeneous macro/net graph plus global geometric branch and scheduled physical guidance | Directly identifies the static-graph limitation in the current typed encoder |
| [EGGPlacer, arXiv:2610.33099](https://arxiv.org/abs/2610.33099) | Recompute analytical standard-cell/full-netlist response during reverse sampling | Provides the strongest mechanism for the p2b-to-routed gap, but its per-step response cost is too high for immediate wholesale transfer |
| [C2FPlace, arXiv:2609.34452](https://arxiv.org/abs/2609.34452) | Partial rip-up/re-place, two-phase reconstruction schedule, critical-macro tuning | Maps to state-preserving local repair after a strong parent exists |
| [DiffPlace, arXiv:2510.15897](https://arxiv.org/abs/2510.15897) | Vector-wise message passing, decoupled global energy and local manifold constraints | Supports separating soft global response from hard outline/overlap projection |
| [SteerGenSE, arXiv:2608.27649](https://arxiv.org/abs/2608.27649) | Partial re-noising as a diffusion mutation kernel, risk-controlled pruning | Supports re-noising selected parts of a legal parent rather than restarting every proposal |
| [PPAPlace, arXiv:2608.13790](https://arxiv.org/abs/2608.13790) | Dual-stream netlist graph plus spatial density/RUDY predictor trained on post-global-route labels | Defines a fallback if a global-place scalar still does not transfer to routed PPA |
| [MacroAgent, arXiv:2608.24946](https://arxiv.org/abs/2608.24946) | Cluster/contour/template regularized legalization | Secondary legalization reference; no model integration until displacement or regularity is the measured failure |

MacroDiff+ reports that removing the hetero-graph branch raises legalization
displacement by 111.7% in its own benchmark, while removing the global geometric
branch raises it by 95.4%. This does not transfer numerically to FloorSet, but
it supports testing a dynamic topology/geometry junction rather than adding
capacity to either branch separately.

EGGPlacer reports that analytical response guidance improves routed
wirelength even when macro-only HPWL is not the best proxy. Its ablation also
reports a negative rank relation between mHPWL and routed wirelength on its
guidance family. This is aligned with our p2b/routed disconnect and justifies a
full-netlist response signal.

PPAPlace reports that post-global-routing labels preserve final timing ranking
better than pre-route timing and that graph and spatial streams are
complementary. We cannot reuse its code because the repository has no explicit
license file. A future implementation would use only the architectural idea and
our own ORFS parser.

## 4. Candidate matrix

### U1: Dynamic state-conditioned graph relay

Priority: P0 architecture candidate.

Pain point: matched-compute convergence plateau and weak transfer from static
b2b/p2b conditioning.

Existing path:

- `src/floorflow/chip_flow/hierarchical_flow.py`
- `TypedGraphEncoder`
- `GraphMessageBlock`
- `DenoiserCond.adjacency`

Missing path: current positions and shapes do not modulate b2b/p2b messages
during sampling.

Minimal extension:

- add one opt-in, zero-initialized state-to-edge geometry relay inside the
  existing graph-message path;
- compute only active-edge quantities such as normalized center delta, projected
  overlap separation, size ratio and net weight;
- aggregate with the existing sparse `index_add_` path;
- do not recompute dense typed attention at every ODE step;
- keep the disabled path bit-compatible with the current checkpoint.

Paper mapping: MacroDiff+ dynamic graph `G_t`; DiffPlace vector-wise message
passing.

Primary falsification:

- no improvement in cost or V_rel on the frozen 2k/refiner400 screen;
- 5k/refiner1600 confirmation improves b2b/p2b only while legalized cost and
  V_rel regress;
- gain disappears when dynamic geometry is replaced by a static A5-style gate;
- operator profiling shows the path breaks the GPU budget.

### U2: Response-guided critical-block partial reflow

Priority: P1, only after M6 validates a deployable response.

Pain point: response ranking may identify better parents, but the model cannot
repair those parents without restarting the full trajectory.

Existing path:

- `sample_hierarchy_flow`
- `scripts/evaluate_chip_analytical_search.py`
- closed-form refiner and hard projector

Missing path: no partial re-noising, rip-up or critical-block reconstruction.

Minimal extension:

- take the best already legal parent from a fixed-budget proposal set;
- select a small critical block set from candidate-independent graph degree,
  current net span and the validated M6 response;
- perturb only those blocks in the existing latent state;
- spend the remaining frozen NFE/refiner budget on the partial trajectory;
- account for legalization and response wall time explicitly.

Paper mapping: C2FPlace partial rip-up/re-place; SteerGenSE partial re-noising;
EGGPlacer response guidance without copying its every-step cost.

Primary falsification: no cost/V_rel improvement at equal total budget, or the
gain exists only in p2b/global-place estimate and not in the certified ORFS
endpoint.

### U3: Outline-aware manifold projection

Priority: P1 extension of M9.

Pain point: fixed-outline feasible fraction remains zero despite L0 outline
conditioning and larger draw budgets.

Existing path:

- M9 `outline_to_hidden`
- exact-area decode
- hard constraint projector
- scale-pack legalizer

Missing path: no deterministic fit-to-outline manifold after exact-area decode.

Minimal extension:

- retain the existing L0 condition;
- after exact-area decode, apply outline-aware translation/scale and overlap
  projection before shelf fallback;
- preserve fixed shapes, MIB sharing, preplaced blocks and area exactly;
- expose whether outline infeasibility, overlap or scale-pack core-fit is the
  failing stage.

Paper mapping: DiffPlace global-energy/local-manifold split; MacroDiff+ scheduled
overlap guidance.

Falsification: conditioning improves but feasibility stays zero; or feasibility
is obtained only by moving fixed/preplaced blocks or changing exact area.

### U4: Dual-stream post-GRT PPA surrogate

Priority: P2 research fallback; data-gated.

Pain point: routed PPA remains negative and even the global-place response may
rank proposals without transferring to final route.

Existing path: M6 direct global-place response only.

Missing path: no standard-cell density, RUDY-style congestion, pin-density or
post-GRT PPA predictor.

Minimal extension:

- only after M6 fails to transfer and a matched label corpus exists;
- use one graph stream over the netlist and one spatial stream over density,
  pin, macro occupancy and RUDY-like channels;
- train with leave-one-design/family-out ranking loss using post-GRT labels;
- inject a gradient only through a projected/trust-region repair stage.

Paper mapping: PPAPlace dual-stream predictor; GOALPlace learned density target;
DAS-MP standard-cell cluster interaction.

Hard gate: no implementation without matched full-netlist labels and a
label-fidelity audit. Do not train a surrogate on our macro-only p2b labels and
call it routed-PPA aware.

## 5. Rejected interpretations

- Do not revive A15/A16 as a new scalar energy.
- Do not run the static A5 edge gate again.
- Do not add a second typed attention layer and call it dynamic graph
  conditioning.
- Do not add a routing, PDN, CTS or STA head without matched labels and a
  concrete placement action.
- Do not compare paper-reported ISPD/MMS numbers with FloorSet/ORFS numbers.
- Do not treat C2FPlace or PPAPlace code as reusable until an explicit license
  is present.
- Do not use post-route outcomes for online candidate selection and then report
  them as a deployable method.

## 6. Execution order after the current blocker clears

1. Finish A17, the M6 response queue and the serialized corrected-PPA queue.
2. Decide whether the global-place response beats p2b on new proposals.
3. If M6 passes, run U2 as a bounded test-time repair and keep U1 as the training
   mechanism candidate.
4. If M6 fails or ranking saturates, prioritize U1.
5. Run U3 only on the frozen fixed-outline cohort after A17 releases compute.
6. Keep U4 dormant until a matched post-GRT label corpus and label-fidelity
   audit exist.
7. Every candidate uses `cuda:0`, the 2k/refiner400 quick subset and
   5k/refiner1600 confirmation before any 20k/full-data run.

## 7. Post-M10 ordering research

Two additional primary sources point to a mechanism that fits the existing
per-block architecture without adding another dense attention block:

- [OrderPlace, arXiv:2606.08904](https://arxiv.org/abs/2606.08904) treats macro
  placement order as a first-class optimization variable. Its proxy-guided
  search finds static and dynamic ordering policies and argues that early
  decisions trigger irreversible downstream constraints.
- [Re2MaP, arXiv:2511.08054](https://arxiv.org/abs/2511.08054) recursively
  prototypes and relocates macro groups, deferring part of the blocks to later
  rounds while using a packing-tree relocation stage.

These observations become relevant only if the current 21-120 range seed
replication is unstable. The minimal architecture response would reuse the
existing L0 boundary, L1 cluster slots, L2 block flow and L3 port state:

1. add a small phase/order embedding to the existing hierarchy state;
2. schedule cluster/block updates in stages or defer low-confidence groups;
3. condition later stages on the already legalized group positions;
4. compare against the simultaneous-flow control on the frozen 20k subset.

This route is not authorized yet. The data-coverage seed result must decide
whether ordering, rather than missing high-block examples, is the unresolved
mechanism. No external source code is copied.

## 8. Source-code reuse and licenses

| Repository | Verified commit | License status | Decision |
|---|---|---|---|
| [MacroDiff-plus](https://github.com/jhy00n/MacroDiff-plus) | `c28ef15bc9ebba6c80eedaf482beb322a7ede4b1` | MIT | Reference dynamic hetero-graph and scheduled guidance patterns; task and data differ |
| [C2FPlace](https://github.com/lxxxxb/C2FPlace) | `b93cc65986440c63b9e907699b1861834f27e25c` | No license file found | Algorithmic reference only; do not copy code |
| [PPAPlace](https://github.com/ValleyC/PPAPlace) | `e07bddc4c258c45bce6d198667406a38ab4f6cd9` | No license file found | Algorithmic reference only; do not copy code or dataset before separate review |

Within this repository, reuse `GraphMessageBlock`, `masked_token_mean`,
`decode_boxes_batch`, `integrate_hierarchy_step`,
`project_hard_constraints`, `scale_pack_legalize_reproducible`,
`summarize_analytical_response` and the Trinity closed-form refiner. New code is
allowed only after recording why these paths cannot express the proposed
operator.

