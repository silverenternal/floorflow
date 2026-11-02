# Chip SOTA Parameter and Compute Budget Audit v1

Date: 2026-10-30

Status: planning audit. Numbers below are either measured from pinned local
artifacts or quoted from primary paper/repository reports. Cross-paper metrics are
not treated as directly comparable unless task, decoder, scorer and legalizer
match.

## 1. Why this audit matters

The current model is not obviously under-parameterized at the exact Trinity d192
comparison. It is substantially under-trained relative to the published Trinity
recipe. This changes the rescue order:

1. first test training-token/compute scaling at the same architecture size;
2. then test inference budget and candidate-selection scaling;
3. then test width/depth scaling;
4. only after those axes are understood, add another architecture mechanism.

An absolute-best result and a compute-efficient result are different claims:

- **absolute SOTA**: lowest error under any declared budget;
- **same-model-budget SOTA**: equal parameter count and training cost;
- **same-inference SOTA**: equal parameters, NFE, refiner steps and samples;
- **equal-quality efficiency SOTA**: fewer NFE, refiner steps, samples or
  GPU-seconds to reach the anchor quality.

Every future table must report all four budget axes. A win obtained only by
increasing budget is not a same-compute win.

## 2. Exact current-model budget

Production typed-hierarchy checkpoint:

`outputs/chip_hierarchy/v7_bbox_anneal_20k/typed_hierarchy_bbox_anneal_seed19_20k/checkpoints/final.pt`

| Item | Current value |
|---|---:|
| Parameters | 5,836,447 |
| Width | 192 |
| Depth | 10 |
| Typed graph layers | 1 |
| Training steps | 20,000 |
| Training micro-batch | 128 |
| Effective batch | 128 |
| Nominal sample exposures | 2.56M |
| Primary inference endpoint | NFE4 + 400 refiner + 1 sample |
| Closest-anchor diagnostic | NFE4 + 1,600 refiner + 1 sample |
| Frozen feasibility | 1.0 |

Measured local wall time for the current NFE4 + 1,600 candidate on Official-100
was about 999 seconds for 100 cases, or 9.99 seconds per case. The matched
Trinity d192 report records 14.67 seconds per case under the same local nominal
protocol. These wall times are machine- and load-dependent, so nominal
NFE/refiner/sample counts remain the primary budget fields.

## 3. Exact Trinity comparison

The pinned Trinity d192 safetensors checkpoint contains 5,856,400 parameters.
The README rounds this to 5.9M.

| Model | Parameters | Training recipe | Published inference recipe |
|---|---:|---|---|
| Trinity d192 | 5,856,400 | 200k steps, effective batch 256 | up to 32 Euler NFE, 400-step refiner, best-of-48 |
| Trinity d256 | 10.6M | same recipe family | same-style budget controls |
| Trinity d384 | 23.3M | same recipe family | same-style budget controls |
| Trinity d512 | 42.1M | same recipe family | same-style budget controls |
| Trinity flagship | 92.9M | 200k steps, effective batch 256 | up to 32 Euler NFE, 400-step refiner, best-of-48 |

README-reported FloorSet validation reference:

- flagship mean hard cost: `1.014`, best of 48;
- reported time: `1.63 s/chip`;
- no directly comparable V_rel value is published for that headline in the
  pinned README.

The parameter ratio is:

- ours vs d192: `5,836,447 / 5,856,400 = 0.9966`, essentially parameter-matched;
- flagship vs ours: `92.9M / 5.836M = 15.9x`.

The nominal training sample-exposure ratio is:

```text
Trinity: 200,000 steps * 256 = 51.2M
Ours:     20,000 steps * 128 =  2.56M
ratio: 20x
```

Because our depth is 10 and d192 depth is 12, realized FLOPs may be slightly
more than 20x in Trinity's favor. Exact FLOP accounting remains a required
artifact.

## 4. Other primary-source budget anchors

These methods use ISPD/IBM-style macro-placement protocols and cannot be compared
numerically to FloorSet/ORFS without a protocol bridge.

| Method | Parameters | Training budget | Inference/search budget | Primary-source caveat |
|---|---:|---|---|---|
| ChipDiffusion Large | 6.29M | 1000-step DDPM setup; repo default 3M updates, batch 64 | 10 guidance steps; paper average 4.39 min/IBM design, 21.2 min/ISPD design | Different task and scoring |
| MacroDiff+ | 5,023,171 from released checkpoint | 200 diffusion timesteps; per-design training details differ | DDIM 20 steps plus 300-700 per-design guidance iterations in paper; 30-91 s/design | Heterogeneous graph and ISPD protocol |
| EGGPlacer | 556,354 | 3M updates, batch 64 | analytical Ksc=5 per reverse step, K=10 inner energy steps, DDPLacer total 95.40 s/design average | Strong physical response but expensive guided inference |
| RollPlace | model plus MCTS | base placer training plus search | 29-156 min/design for RollPlace variants on ISPD table | Search budget dominates; not a pure parameter comparison |
| C2FPlace | non-neural evolutionary search | no learned parameter count | population search plus critical-macro tuning | Refinement/search method, not generator backbone |

The main takeaway is that published methods do not share a budget unit. Some use
tiny backbones with expensive physical/search loops; others use larger denoisers
with short inference. Reviewers will accept budget scaling only when every axis
is disclosed.

## 5. Parameter ladder for our architecture

Counts below are instantiated with the current typed-hierarchy architecture,
`depth=10`, `typed_layers=1`, `head_dim=64`, and valid head counts:

| Variant | Width | Heads | Parameters | Ratio to current |
|---|---:|---:|---:|---:|
| d192-current | 192 | 3 | 5,836,447 | 1.00x |
| d256 | 256 | 4 | 9,844,383 | 1.69x |
| d384 | 384 | 6 | 20,219,551 | 3.46x |
| d512 | 512 | 8 | 35,215,007 | 6.03x |
| d768-l12 | 768 | 12 | 86,445,345 | 14.81x |

Do not jump directly to d768. Parameter scaling must follow a matched
small-subset curve and must be compared at:

- equal optimizer steps;
- equal sample/token exposure;
- equal approximate FLOPs;
- equal inference budget.

The production typed-hierarchy runtime preflight (40 blocks, batch 1,
NFE4, CUDA:0) gives:

| Width | Parameters | Train forward+backward FLOPs | Sample NFE4 FLOPs | Peak GPU memory |
|---:|---:|---:|---:|---:|
| 192 | 5,836,447 | 1.421B | 1.786B | 112 MB |
| 256 | 9,844,383 | 2.358B | 2.959B | 144 MB |
| 384 | 20,219,551 | 4.771B | 5.956B | 226 MB |

These are single-batch preflight measurements, not quality or throughput
results. They establish the normalization factors needed before the S3
capacity ladder can run.

The same preflight at the distribution extremes gives:

| Blocks | d192 train FLOPs | d256 train FLOPs | d384 train FLOPs |
|---:|---:|---:|---:|
| 21 | 0.763B | 1.278B | 2.608B |
| 40 | 1.421B | 2.358B | 4.771B |
| 120 | 5.061B | 8.244B | 16.515B |

The measured wall-clock values at batch 1 are dominated by fixed dispatch
overhead, so they are not used as equal-quality efficiency evidence.

At batch 32, 40 blocks, the training forward+backward FLOPs are `46.872B`,
`77.951B` and `158.209B` for d192/d256/d384, with peak allocations of roughly
465/593/836 MB. Short-run wall times remain unstable across width order and are
not used as efficiency evidence.

The pre-registered S3 screen now separates two budget classes at the 20k
confirmation horizon:

| Arm | Steps | Budget meaning |
|---|---:|---|
| d192 control | 20,000 | reference |
| d256 equal exposure | 20,000 | same samples, ~1.65x FLOPs |
| d256 equal FLOPs | 12,089 | approximately equal training compute |
| d384 equal exposure | 20,000 | same samples, ~3.35x FLOPs |
| d384 equal FLOPs | 5,978 | approximately equal training compute |

A 5k/2k-style quick screen uses the same relative reduction before the 20k
confirmation. D384 remains gated on D256 improving both primary metrics in at
least one budget class.

`scripts/run_chip_capacity_screen.py` implements the gated sequence directly:
it waits for S1, runs the d192 control, both d256 budget classes, and opens only
the matching d384 budget class when d256 passes the two-metric gate. All
evaluations use frozen subsets and the common evaluator.

A one-step formal-CLI smoke additionally confirmed that d256 and d384 can build,
finish one forward/backward step, save EMA checkpoints, and reload with the
expected `9,844,383` and `20,219,551` parameters. This is plumbing evidence only,
not a convergence result.

## 6. Recommended attack order

### Route A: training-compute scaling at d192

This is the cheapest and most evidence-backed first experiment because d192 is
already parameter-matched to the anchor:

- frozen 20k-case subset first;
- effective batch 256 via accumulation of two micro-batches of 128;
- steps: 20k, 50k, 100k, 200k;
- compare at fixed NFE4/refiner400 and at the closest-anchor NFE4/refiner1600;
- stop before the next step count if both primary metrics fail to improve.

Only the winning horizon may enter the final full-data run.

The maintained trainer now supports this directly through
`--grad-accum-steps 2 --batch-size 128`. A 1-step CPU smoke verified the reported
effective batch, nominal sample exposure, checkpoint configuration and
balanced-sampler cursor.

### Route B: inference-budget scaling

Use one frozen checkpoint and change one axis at a time:

- NFE: 4, 8, 16, 32;
- refiner: 400, 800, 1600, 3200;
- samples: 1, 4, 16 using a deployable shared scorer;
- oracle best-of-N remains a diagnostic and cannot support a deployed SOTA claim.

The primary NFE4/refiner400 endpoint remains separate. This route answers whether
the model's ceiling or its converged sample quality is the limiter.

An 8-case development probe first found monotonic improvement from refiner400
through refiner3200. On the predeclared canonical Official-8 set, however, the
matched comparison did not support a SOTA claim:

| NFE4 + 3200 refiner | Candidate | Trinity d192 |
|---|---:|---:|
| Cost | 1.23492 | 1.11253 |
| V_rel | 0.02248 | 0.03315 |
| Feasibility | 1.0 | 1.0 |

The candidate cost is `+11.00%` and V_rel is `-32.19%` relative to the anchor;
both paired case intervals cross zero. The failure is dominated by the 120-block
case, where candidate cost is `2.668` versus Trinity `1.131`. This retains a
high-complexity failure signal and rules out treating extra refiner budget alone
as the SOTA solution.

### Route C: capacity scaling

After Route A identifies the best training horizon:

- screen d256 on the frozen 20k subset;
- only if d256 clearly improves scaling, test d384;
- d512/d768 are full-compute candidates, not exploratory defaults;
- compare at equal sample exposure and approximate FLOPs, not only equal steps.

### Route D: data and architecture

Run these only after A-C establish a clean budget frontier:

- 50k -> 200k data scaling at the selected training horizon;
- U1 dynamic state-conditioned graph relay;
- U2 bounded analytical-response repair;
- combination only if both mechanisms pass independently.

### Route E: high-block-count coverage

The maintained 50k cache used by the current model is limited to 21-60 blocks,
while the Official-100 evaluation extends to 120 blocks. The Official-8 result
shows the expected failure: the 120-block candidate cost regresses sharply.

Frozen range-scaling subset:

`configs/chip_train_scaling_range20k_v1.json`

- 20,000 unique cases;
- block counts 21-120;
- selection seed `20261031`;
- zero dev overlap;
- source cache
  `outputs/cache/chip_floorset_200k_train_12k_dev_range_v1/train`.

This is a separate data-coverage axis. Compare the same d192 training horizon on:

1. the original 21-60 subset;
2. the 21-120 range subset.

If the 120-block outlier closes without regressing low/mid-complexity cases, the
gap is a training-distribution issue rather than an architecture-capacity issue.

The high-block Official-8 diagnostic confirms the gap:

| NFE4 + 3200 refiner, blocks 61-120 | Candidate | Trinity d192 |
|---|---:|---:|
| Cost | 1.27154 | 1.03772 |
| V_rel | 0.03983 | 0.01720 |
| Feasibility | 1.0 | 1.0 |

Candidate cost is `+22.53%` and V_rel is `+131.54%` relative to the anchor.
The candidate is close on 61/69/78-block cases, mixed at 95, and degrades
sharply at 103/112/120 blocks. This evidence prioritizes training on the full
21-120 block range before attributing the remaining SOTA gap to width or depth.

The same high-block cases at refiner1600 materially narrow the gap:

| NFE4 + 1600 refiner, blocks 61-120 | Candidate | Trinity d192 |
|---|---:|---:|
| Cost | 1.07141 | 1.05387 |
| V_rel | 0.01917 | 0.01878 |

Candidate cost is only `+1.66%` and V_rel `+2.08%`; both paired intervals cross
zero. Therefore the 3200-step result is not a monotonic budget effect: Trinity
continues improving, while the current candidate over-refines and creates
outliers. Refiner budget needs a trust region or adaptive stop before it can be
used as a SOTA lever.

## 7. SOTA certification

Three anchors are required:

1. exact Trinity d192 at matched model/training/inference budget;
2. exact Trinity flagship 92.9M or the strongest available public same-protocol
   model at a declared absolute budget;
3. the current model at a frozen budget ladder covering single-sample,
   higher-NFE/refiner and deployable best-of-N points.

The absolute table must include:

- parameters;
- training steps, batch and sample exposures;
- NFE;
- refiner steps;
- draws/best-of-N;
- legalizer and scorer;
- wall time and GPU;
- legalized cost, V_rel, feasibility, and external routed PPA guards.

## 8. Primary sources

- Pinned Trinity code and release:
  `third_party/Trinity/README.md`,
  `third_party/Trinity/docs/training.md`,
  `third_party/Trinity/docs/sampling.md`,
  `third_party/Trinity/checkpoints/d192/config.json`.
- ChipDiffusion:
  <https://arxiv.org/abs/2407.12282>,
  <https://github.com/vint-1/chipdiffusion>.
- MacroDiff+:
  <https://arxiv.org/abs/2605.16451>,
  <https://github.com/jhy00n/MacroDiff-plus>.
- EGGPlacer:
  <https://arxiv.org/abs/2610.33099>.
- RollPlace:
  <https://arxiv.org/abs/2610.06316>.
- C2FPlace:
  <https://arxiv.org/abs/2609.34452>,
  <https://github.com/lxxxxb/C2FPlace>.

## 9. Competitive position refresh

The current model should not be treated as a 5.8M experiment that merely
needs a larger backbone. It has several properties worth exploiting:

- it is effectively parameter-matched to Trinity d192 (`5,836,447` versus
  `5,856,400`);
- it has seen only `2.56M` nominal training sample exposures versus `51.2M`
  for d192;
- the maintained training cache covers 21-60 blocks while Official-100 reaches
  120 blocks;
- legalized feasibility is `1.0` on the frozen FloorSet comparison;
- NFE4 is already operational and per-block typed graph conditioning is an
  explicit architectural mechanism rather than a scalar post-hoc loss;
- the high-block refiner1600 development slice is within `+1.66%` cost and
  `+2.08%` V_rel of d192, whereas refiner3200 over-refines the candidate.

That combination suggests a stronger route than immediate width scaling:
training exposure and block-count coverage first, bounded inference policy
second, capacity third. Extra compute is allowed only when the exact same
protocol can report both an absolute-budget point and a normalized-budget
comparison.

The public landscape is heterogeneous:

| Frontier | Representative | Learned budget | Deployment/search cost |
|---|---|---:|---:|
| Parameter-matched generative | Trinity d192 | 5.86M, 51.2M exposures | up to 32 NFE, 400 refiner, best-of-48 |
| Large generative | Trinity flagship | 92.9M, 51.2M exposures | same-style budget; 1.63 s/chip headline |
| Medium generative | ChipDiffusion Large | 6.29M, repo 3M updates | 10 guidance steps; minutes/design |
| Small model + physical search | MacroDiff+ | 5.02M checkpoint | DDIM20 + 300-700 guidance iterations |
| Tiny model + heavy analytical loop | EGGPlacer | 0.56M | analytical/energy inner loops; about 95.4 s/design |
| Search-dominated | RollPlace / C2FPlace | model or non-neural search | tens of minutes to population search |

A publication can therefore be built around more than one winning axis:

1. beat the exact d192 anchor after closing its training/data deficit;
2. retain fixed refiner1600 as the safe control after adaptive-stop/search
   policies failed to beat it on development data;
3. show a capacity curve that separates data coverage from model width;
4. certify one absolute-budget point against the flagship or strongest exact
   same-protocol anchor;
5. retain routed-PPA transfer as an independent guard rather than inferring it
   from HPWL.

## 10. Execution order

`S1` trains the unchanged d192 model on the frozen 21-60 and 21-120 ladders at
matched exposure. The adaptive-refiner route is closed after the dev32 audit:
fixed 400/800 lose quality, fixed 3200 trades a 0.10% aggregate cost gain for a
6.43% V_rel gain but regresses 11/32 cases on cost, and neither observable-rank
nor leave-one-case-out selector beats fixed 1600. `S3` then screens d256/d384
under equal-exposure and approximately equal-FLOP controls and certifies the
strongest exact same-protocol anchor.

The 3200-step result is retained as a mixed negative/quality-budget point.
Future adaptive refinement may reopen only with a new pre-generation or
EDA-response signal that independently clears a total-compute Pareto gate.
Oracle snapshot selection remains diagnostic and cannot support a deployment
claim.

## 11. Live SOTA-frontier refresh (2026-10-30)

This refresh rechecked the primary repository and paper sources. It changes the
planning conclusion from "try a larger model" to "separate five budgets and
attack the cheapest unresolved axis first".

| Model / method | Learned budget | Inference or search budget | Protocol position |
|---|---|---|---|
| Current typed per-block flow | 5,836,447 parameters; 20k steps, batch 128, 2.56M exposures | NFE4 + 400 or 1,600 refiner, one sample; about 9.99 s/case at 1,600 locally | Exact current candidate |
| Trinity d192 | 5.856M; 200k steps, effective batch 256, 51.2M exposures | up to NFE32 + 400 refiner; best-of-48 headline | Exact same-protocol anchor |
| Trinity flagship | 92.9M | same-style NFE/refiner/search; README reports 1.014 hard cost at 1.63 s/chip | Absolute-budget anchor |
| FlowPlace (DAC'26) | Comparable learned-parameter count not reported | 20-50 Euler steps; released test uses 20 plus hard-constraint sampling | Closest external full-OpenROAD PPA anchor |
| ChipDiffusion Large | 6.29M | 10 guidance steps; minutes/design | Different benchmark |
| MacroDiff+ | 5.023M | DDIM20 + 300-700 guidance iterations; 30.4-90.9 s/design | Different benchmark |
| EGGPlacer | 0.556M | Ksc=5 analytical and K=10 energy steps; 95.40 s/design | Tiny generator + heavy response |
| RollPlace | No comparable standalone parameter budget | MCTS/rollout; tens to >150 min/design | Search-dominated |
| C2FPlace | Non-neural search | population search; paper uses a 1,000-minute budget | Search-dominated |
| PlaceReasoner-Beta | Qwen3-VL-8B/30B or undisclosed Claude backbone | five candidates + checkers/optimization; 602-848 s/design and 176k-386k tokens/design | Agentic end-to-end context |

The current model is therefore not obviously parameter-starved against d192:
its parameter ratio is `0.997x`, while its nominal exposure ratio is only
`0.05x`. It is 15.9x smaller than the Trinity flagship. The cheapest credible
absolute-SOTA attempt is not to jump to d768; it is to close the 20x
training/data gap, then ask whether a 1.65x d256 model improves the converged
ceiling. Inference/search scaling is a separate route.

The updated portfolio has seven independent frontiers:

1. **R1 training and data coverage (P0).** Run the frozen 20k/50k/100k/200k
   ladder and the 21-120 block-count branch first.
2. **R2 capacity (P1).** Compare d256 at equal exposure and approximately
   equal FLOPs; open d384 only after d256 improves both primary metrics.
3. **R3 inference and samples (P1).** Sweep NFE, refiner steps and 1/4/16/48
   draws. Oracle best-of-N is only a ceiling; deployable best-of-N requires a
   frozen label-free selector and full accounting.
4. **R4 architecture (closed negative).** U1's state-to-edge relay is rejected
   at confirmation. The clean-room projection-consistent/final-only family was
   then tested without new parameters. A quick refiner400 gain reversed at
   refiner1600 in both primary metrics, so M10 is rejected without full data.
5. **R5 response-guided search (P1).** Open only after M6 demonstrates
   incremental ranking value beyond p2b on untouched proposals.
6. **R6 routed-PPA transfer (P0).** A FloorSet win remains a structural result
   until routed WL, congestion, timing, power and DRC guards pass on a fresh
   set.
7. **R7 public-budget certification (P1).** Report Trinity d192, Trinity
   flagship and the FlowPlace-style end-to-end frontier under named
   parameter/training/inference budgets. Parameter efficiency for FlowPlace
   cannot be claimed because its comparable count is unpublished.

Operator work is part of the budget frontier, not a separate quality claim.
Safe candidates are caching invariant edge indices/degrees/typed masks,
fusing repeated masked reductions, reducing launch and `nonzero` overhead, and
benchmarking fused optimizer/EMA or compiled stateless subgraphs. A speedup is
accepted only after exact forward/gradient or final legalized-metric
equivalence. The compiled projector remains non-default because its small
stepwise drift changed full trajectories.

The public sources checked in this refresh are
[Trinity](https://github.com/Kohaku-Lab/Trinity),
[FlowPlace](https://arxiv.org/abs/2604.23658)
([official code](https://github.com/lamda-bbo/flowplace)),
[ChipDiffusion](https://arxiv.org/abs/2407.12282),
[MacroDiff+](https://arxiv.org/abs/2605.16451),
[EGGPlacer](https://arxiv.org/abs/2610.33099),
[RollPlace](https://arxiv.org/abs/2610.06316),
[C2FPlace](https://arxiv.org/abs/2609.34452), and
[PlaceReasoner-Beta](https://arxiv.org/abs/2609.21263). Cross-paper
parameters and runtimes establish the design space; they do not substitute for
an exact local same-protocol run.

