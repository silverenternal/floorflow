# Chip Architecture Gap Closure v255-v256

Date: 2026-11-02

This review covers the FloorSet chip placement path only. It makes no GPU,
quality, routed-PPA, or SOTA claim.

Verification after this pass: `660` chip tests pass, scoped Ruff passes, and all
modified chip Python files remain below 600 lines. The atom-isolation audit
reports zero unowned and zero duplicated mainline non-backbone parameters.

## Current Standing

The deployed evidence candidate is still the 6,797,217-parameter S3 structural
selector. It has positive pooled continuous diagnostics, but it does not yet
beat the reproducible Trinity d192 anchor on legalized cost and V_rel.

The corrected mainline candidate is a separate, unvalidated architecture:

- 7,209,746 parameters including four endpoint-mode candidates and the typed
  trajectory atom aggregator;
- Trinity d192 reference: 5,856,400 parameters;
- current mainline parameter overhead: `+23.11%`;
- the extra capacity must therefore be justified by matched ablations, not by
  parameter count or proxy wins.

## Metric-Owned Architecture Gaps

| Weak metric / reviewer concern | Network or runtime path | Defect found in v255 | How this revision answers it |
|---|---|---|---|
| b2b HPWL and high-degree tails | static typed graph + `StateEdgeRelay` | FloorSet b2b rows are undirected, but the typed message block defaulted to a one-way source-to-target message while degree normalization counted both endpoints. Directly enabling a reverse message also changed warm-start outputs abruptly. | Mainline enables a zero-gated reverse-message path. At initialization it is exactly the old directional model; after training it can use both graph directions. The dynamic state relay remains the per-ODE geometry path. A matched corrected-mainline test is still required. |
| dynamic graph gate reachability | `dynamic_graph_input` | Both the relay gate and its 3-D projection were initialized to zero, creating a two-zero multiplicative path: the exact parent output was preserved, but the relay gate could not receive a first-step gradient. | The relay gate remains zero while its 3-D bridge uses a small random projection. The model output is still identical at initialization, and a full-model test verifies that the gate opens without a deadlock. |
| grouping and candidate coverage | L1 cluster/MIB/boundary tokens, group moments, shape modes | The structured hierarchy head existed but the mainline config did not enable the structured-head suite; membership hierarchy and standalone legalizer paths were duplicated. | Mainline now uses one structured-head suite for hierarchy coupling, factorized center/rho, dual semantic ports and legalizer response. Duplicate membership/legalizer paths are disabled. |
| L1 relation causality | structural relation head and L1 output | The relation head updated structural tokens after the L1 prediction had already been computed, so relation evidence could not influence the output it was meant to supervise. | Relation tokens now modulate L1 memory before the L1 head. A matched test verifies both the output change and first-step gradient reachability. |
| endpoint candidate learning | shape-mode generator and Pareto selector | Training described the baseline with the current interpolant, while inference described it with the generated endpoint. At zero-initialized residuals every mode tied the baseline, so candidate-WTA selected baseline and the generator head received no residual gradient. Rho-only modes also could not create grouping/b2b/p2b candidates. | Descriptor computation now uses the predicted endpoint in training, matching sampling. A separate mode-only WTA term trains candidates even when the conservative baseline wins. Modes can now move soft centers within a bounded radius as well as change rho; preplaced centers remain fixed. |
| inactive structural slots | native group-moment rollout | Native moment prediction remained nonzero on inactive cluster/MIB/boundary slots, while their supervised targets were zero. This leaks state and dilutes the moment loss. | Native moment predictions are now multiplied by the active structural mask before entering the next state. |
| area / outline / rho | exact-area decoder + factorized block head + L0 envelope | Center and rho were previously mixed in one base output head; the factorized head and same-call global-envelope consensus were not active in the mainline candidate. | Factored center/rho heads are enabled with a bounded rho trust radius, and the bounded global-bbox consensus now consumes the same-call L0 forecast. Global envelope remains separate from per-block shape. |
| p2b and port-to-route transfer | L3 port head -> dual semantic ports -> port consensus -> L2 | `DualSemanticPortHead` ran after `PortConsensusCoupler`, so port consensus consumed the unrefined L3 prediction. Refined ports could not influence the block residual. | Dual semantic ports now run before consensus. The consensus residual consumes the actual refined port geometry. |
| cost, V_rel and legalizer displacement | refiner -> hard projector -> DBU endpoint | The mainline model configuration did not explicitly enable the hard projector; endpoint-specific weights existed without the parent endpoint-compatibility objective. The legalizer response loss also trained on preplaced and padded blocks, and an unrestricted final action damaged the first matched screen. | Mainline explicitly enables the conflict-aware hard projector and a delayed legalizer-aware endpoint objective. Response supervision is masked to valid movable centers. The final action is default-off and now has pin/net-relative per-block trust radii if re-enabled. |
| long-budget convergence | x0 flow + legalizer action | The legalizer response head was built but its apply path was disabled in the mainline candidate. | Final-step bounded legalizer action is enabled. It starts at identity because the displacement head is zero initialized. |
| short-NFE training | NFE consistency rollout | The rollout rebuilt `HierarchyState` from five constructor fields and silently dropped the native seven-dimensional group-moment state. With mainline group moments enabled, the next model call no longer received the state it was trained to refine. | NFE consistency now uses dataclass replacement, preserving moment and previous-state fields. A group-moment gradient test covers the failure. |
| route transfer | route demand conditioning + training proxy | The training proxy gained L-path macro obstruction in v244, but route-aware occupancy conditioning still encoded only midpoint demand. The optimizer therefore knew about blocked L paths while the network input did not. | Route-aware occupancy now computes the same differentiable two-L-path obstruction signal for b2b and p2b edges and injects a per-block obstruction/degree feature through a zero-initialized encoder. Old checkpoints remain exact until training opens it; external routed-WL transfer still requires fresh confirmation. |

## Configuration Coherence

The mainline now has one owner for each hierarchy function:

```text
L0/L1 same-call routing     -> structured_heads.hierarchy_coupling
L2 center/rho factorization -> structured_heads.factorized_block
L3 connected/unconnected    -> structured_heads.dual_semantic_ports
legalizer response           -> structured_heads.legalizer_response
hard final action            -> opt-in, pin/net-radius gated; default off
hard geometry               -> conflict-aware projector
endpoint candidates          -> baseline + four bounded center/rho hypotheses
trajectory atom aggregation  -> typed sparse attention over seven branch residuals
```

`membership_hierarchy_conditioning` and `standalone_legalizer_response` are
disabled in this composition because they duplicate structured-head functions.
This removes dormant parameters and ambiguous ownership.

## Resume and Objective Contracts

The resume manifest now fingerprints all new metric-owned schedules:

- grouping proxy weight, delay and anneal;
- routing proxy weight, delay and anneal;
- legalizer response weight, delay and anneal;
- displacement and rung weights;
- endpoint compatibility weight, iterations, refiner trajectory and auxiliary
  geometry terms.

Without these fields, a resumed run could silently change the objective that
produced its checkpoint.

## Required Ablations

## First Matched Screen and Structural Response

The preregistered 500-step screen used the canonical M27 seed21 initialization,
a disjoint 1k training subset and 32 dev cases at NFE4/refiner400.
It failed the primary gate:

| Metric | Relative change vs parent |
|---|---:|
| legalized cost | `+12.50%` |
| legalized V_rel | `+22.57%` |
| legalized b2b gap | `+101.24%` |
| legalized area gap | `+94.92%` |
| legalized p2b gap | `+8951.55%`, dominated by three large cases |
| grouping | `+6.90%` |
| boundary | `+28.21%` |
| rho MAE | `+6.55%` |

The failure is retained in
`outputs/reports/chip_s0_corrected_screen_v1.json`. Diagnostics:

- the endpoint selector chose baseline on `32/32` cases, so endpoint modes did
  not cause the measured regression;
- raw geometry also regressed, so this is not solely a legalizer artifact;
- an action-only ablation was started and then stopped under the explicit
  no-more-GPU-experiment instruction. No isolated attribution is claimed.

Structural response:

- final legalizer action is now default-off;
- if re-enabled, displacement is bounded by 10% of nearest pin distance and
  25% of nearest net distance in addition to the global trust radius;
- MIB rho sharing uses the deterministic masked-median operator;
- the coupled all-channel training schedule is replaced by an atom-isolated
  sequence: area -> b2b -> port/route -> boundary -> tail -> fusion;
- atom isolation is enforced by the training validator: every atom must appear
  exactly once, shared/all scopes are rejected, and the final stage must be
  `fusion` only, leaving atom heads and shared channels frozen during
  cross-attention aggregation;
- every non-backbone parameter in the mainline now has an explicit atom or
  fusion owner, including L0/L1/L3 heads, local geometry, channel bridges and
  time embedding; ownership is unique, including the coordinate gate and global
  bbox consensus, which are assigned to `area` only;
- atom mixing is bounded to `[-0.5, 0.5]`, so every branch scale stays positive
  and no atom can invert another atom's residual;
- the aggregator is exactly block-permutation equivariant after fusion gating is
  enabled;
- per-block atom flattening now permutes to `B x block x atom` before attention;
  the previous direct reshape mixed different blocks' atom tokens;
- an explicit branch-availability mask excludes inactive atoms from both the
  residual sum and softmax, so zero-capability branches cannot dilute active
  atom probabilities;
- fusion diagnostics now expose active-branch fraction, gate maximum, minimum
  positive gate and attention entropy for collapse detection;
- a case with no available atom forces `gate_min_positive=0`, keeping all
  diagnostics finite and JSON-safe;
- aggregator diagnostics are propagated through `HierarchyPrediction`, so the
  same values are available to training, sampling and evaluation instead of
  being discarded inside the adapter;
- scalar fusion diagnostics are merged into the training loss-term report; the
  full per-block gate tensor remains available through the prediction object
  without breaking scalar logging;
- old structured-head checkpoints load the new aggregator as a zero-gated
  submodule instead of failing the shared `structured_heads.*` prefix check;
- branch promotion now requires an independent frozen-base screen.

No mainline branch is promoted until the following CPU contract and matched GPU
subset tests pass:

| Ablation | Primary evidence | Falsification |
|---|---|---|
| symmetric vs one-way typed b2b | legalized b2b HPWL and b2b-tail cost | symmetric messages improve only raw proxy or regress cost |
| structured suite on/off | area, rho, p2b, cost, V_rel | structured components do not improve their owned metrics |
| refined-port ordering on/off | p2b and external routed-WL sign agreement | ordering has no effect or harms legalization |
| legalizer final action on/off | legalized cost/V_rel, displacement, rung | action increases solver work without primary gain |
| projector on/off | hard feasibility, displacement, cost tail | projector only changes raw geometry |
| route-obstruction conditioning on/off | routed-WL rank agreement on fresh designs | proxy improves without external transfer |

## Claim Boundary

Still forbidden:

- FloorSet or routed-PPA SOTA;
- a Trinity victory before matched legalized confirmation;
- MIB, opening-specific, or highest-complexity gains;
- treating p2b HPWL or route proxy improvement as routed-WL improvement;
- reporting any unvalidated mainline configuration as a result.

## v257 Metric-Aligned Structural Repair

This pass was CPU-only. It does not promote the rejected 32-case screen and
does not support a Trinity, FloorSet, routed-PPA, or CVPR SOTA claim.

| Weak metric | Defect found | Structural repair |
|---|---|---|
| Legalized cost / V_rel / area / b2b / p2b | Inactive atoms still participated as keys and values in atom-to-atom attention before the branch mask was built. | Availability now masks both attention axes; an inactive atom token cannot alter active-atom outputs or fusion diagnostics. |
| All metric-tail failures | Fusion weights were constrained to `0.5x-1.5x`, so a harmful atom could never be rejected even when its owned metric regressed. | The aggregator has a per-block, per-atom reliability gate initialized to exactly one. Fusion can now attenuate an atom toward zero while preserving the old checkpoint's exact residual sum at initialization. |
| V_rel, grouping, MIB and boundary | The training surrogate did not implement FloorSet/Trinity soft severity: grouping and MIB were case-level scalars, boundary was proportionally scaled, and grouping used the wrong pair-gap rule. | `differentiable_soft_severity` now matches Trinity's grouping, MIB, boundary, `N_soft` and `sev_rel` definitions. A parity test compares the torch implementation with Trinity's reference function. |
| Grouping proxy | The training proxy used `max(gap_x, gap_y)` rather than Trinity's smallest-axis rectangle gap. | Both severity and grouping proxy now share `differentiable_pair_gap` from `batched.py`; no duplicate formula remains. |
| Grouping / short-NFE | Inactive group-moment slots diluted the moment loss, and the NFE rollout preserved the current moment state instead of advancing it. | Moment supervision is masked to active structural slots, and the rollout now advances moments with the same ODE update as the hierarchy state. |
| Shape-mode selection | Training scored an unprojected raw endpoint while sampling scored the condition-projected baseline. | Training now applies the same fixed/preplaced/MIB-conditioned projection before generating descriptors and baseline logits; the training objective code moved into the existing objectives module to keep the head below 600 lines. |
| p2b / routed signal | Spatial occupancy normalized b2b and p2b fields by one shared maximum, while the training proxy combined their demands and path obstruction before normalization. | Both paths now normalize demand and compute obstruction independently for b2b and p2b, expose family-specific values, and combine only after each family has its own unit scale. |
| MIB morphology | The physics and endpoint MIB terms compared only log aspect ratio; `1x2` and `2x4` blocks could have zero MIB penalty despite different required shapes. | Both terms now check log-width and log-height deviations from the group mean. Same-aspect, different-size regression tests fail under the old implementation. |
| Projector-aware endpoint loss | The endpoint-compatibility head projected under `no_grad`, so MIB, pin-tail, projected-area and legalizer-target terms depended only on detached geometry and supplied zero generator gradient. | The projected endpoint is now straight-through, and sidecar displacement uses an explicitly differentiable projector path. Dedicated tests verify nonzero MIB and legalizer-target gradients through the projector-aware path. |
| Outline conditioning | Missing outline state was silently derived from `batch["boxes"]`, which is the target layout. | Training and sampling now fail closed unless an explicit target-free outline state is supplied. The regression test asserts that GT fallback is rejected. |
| Graph conditioning target leakage | Some graph features are constructed from `batch["boxes"]`. | An invariance test perturbs every free block target box and verifies conditioning features, adjacency and mask are exactly unchanged; only fixed/preplaced target fields are allowed inputs. |
| Grouping proxy schedule | Configuration and validation referenced `grouping_proxy_final_weight`, but training read only the initial zero weight. | Routing and grouping now share `scheduled_proxy_weight`; a step-350 real-batch probe confirms both current weights are `0.01` instead of grouping remaining zero. |
| Grouping objective wiring | The JSON objective and resume fingerprint contained `grouping_proxy_final_weight`, but `train_cli.py` had no matching flag. | The CLI exposes the final-weight argument, and a parser regression test verifies the value reaches training arguments. |
| Boundary V_rel | The isolated boundary atom owned only `typed.constraint_type`, while the existing boundary-only output coupler was disabled. | The mainline now enables `BoundaryAnchorCoupler`; it edits only boundary-coded blocks and is owned solely by the boundary atom. |
| Hard legality / raw geometry | Boundary, factorized, global-envelope and trajectory-atom residuals could alter preplaced centers or fixed shapes before the projector repaired them. | Residual heads now preserve preplaced center/rho and fixed rho directly. The learned path stops spending capacity on coordinates that the protocol must reject and then undo. |
| Atom-stage optimization | Every atom inherited one global cosine LR; fusion started at step 750 with an almost exhausted learning rate and never opened properly. | `ChannelStageController` now supports stage-local warmup/cosine restarts. A step-800 fusion probe reaches LR `4.52e-5` instead of the previous near-zero tail. |
| Boundary executor self-harm | The learned boundary residual could move a coded block farther from the predicted envelope. | The coupler now accepts a block residual only when it strictly lowers target-free boundary error. Locked preplaced/fixed degrees remain invariant. |
| S1 confirmation | The first 1000-step quick run failed before stage-local LR and conservative boundary acceptance; the post-refiner projector repair also failed. | The preregistered 1600-step confirmation gives mean legalized cost `-3.76%`, V_rel `-20.56%`, b2b `-43.05%`, p2b `-100.92%`, area `-17.35%`, boundary `-30.43%` and rho `-7.66%` versus the matched control. P2b, boundary and rho intervals exclude zero; cost/V_rel intervals still cross zero. |
| Cost / V_rel / grouping / route | Metric objectives started at step 1000 while atom stages ended at step 750, making their gradients structurally absent from the intended optimization window. | Official/area starts at 25, grouping/b2b at 150, routing/port-route at 300 and legalizer/tail at 600. Training validation rejects objectives that start outside their owned atom window. |

Verification for this pass:

- `671` chip tests pass;
- Ruff passes for every changed Python file;
- no chips-owned Python file reaches 600 lines;
- the atom ownership audit reports `0` unowned and `0` duplicated non-backbone
  parameters;
- the atom self-eval reports `42` contracts, `8/8` runtime probes passing;
- a seeded real-batch probe covers four cases in each of the 21-40, 41-60,
  61-80, 81-100 and 101-120 strata from the existing 21-120 training cache;
- every stratum has a finite loss and finite gradients;
- the same probe passes at step 350, where official-cost, grouping and
  port-route objectives are active;
- the first S1 quick gate is retained as a negative; stage-local LR plus
  conservative boundary acceptance then produce mean legalized cost `-4.40%`
  and V_rel `-4.08%` against the matched 1000-step control, with B2B `-41.30%`,
  P2B `-95.77%`, area gap `-25.48%` and solver time `-23.80%`;
- the S1 primary intervals still cross zero and grouping mean regresses
  `3.85%`, so confirmation remains mandatory;
- a shared post-refiner hard projector is rejected: cost `+4.55%`, area
  `+79.14%`, P2B `+1552%`, with no legalized boundary gain;
- the 1600-step S1 confirmation passes its mean gate with legalization
  feasibility `1.0`; S2 must now compare this frozen endpoint against
  reproducible Trinity d192 and a fresh case-disjoint cohort.
- the dynamic-graph residual mean rises from `0.035` on 21-40 blocks to
  `0.399` on 101-120 blocks in the activated structural probe, providing a
  concrete high-complexity signal to optimize rather than a parameter-count
  guess.

The existing `chip_floorset_200k_train_12k_dev_range_v1` cache already supplies
200k training cases over 21-120 blocks. Future complexity-tail training must use
that cache. Its v2 audit confirms all 100 block counts, a 200k train / 12k dev
split, zero train-dev overlap, and an unchanged reference dev list. Splitting
cases out of dev would invalidate the intended case-disjoint confirmation
protocol.

The 50k training audit also checked every MIB group: all `50,000/50,000`
groups have identical target areas. Under the exact-area decoder, sharing rho
therefore preserves the official shared `(w,h)` requirement rather than only a
weaker aspect constraint.

The earlier degree-conditioned expert and complexity-slot rejections are not
clean high-complexity conclusions: both adapters were trained on the 21-60
cache while their 81-120 strata were evaluated. They remain invalidated as
promotions, but must be retrained as separate range-cache arms before the
complexity hypothesis is rejected again.

These repairs make the next matched screen scientifically interpretable. They
do not establish that the resulting model is better.
