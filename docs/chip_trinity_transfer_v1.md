# Chip Trinity Backbone Transfer v1

Date: 2026-10-31

## Mechanism

The typed hierarchy already reuses Trinity's `SetTransformerDenoiser`, but prior
training initialized it from scratch. M27 strictly initializes the depth12
backbone from the pinned `third_party/Trinity/checkpoints/d192` release, then
trains the existing typed L0-L3 hierarchy and graph encoder.

This is an architecture/transfer experiment, not a new backbone:

- strict backbone load: 5,856,400 parameters;
- total candidate: 6,797,217 parameters;
- release SHA-256:
  `017cc7ecf835f1eb8c03039e7896844181ef5546b52f86fc34b615b9ddd8cdec`;
- typed encoder, hierarchy heads and exact-area flow remain trainable;
- the matched control is the depth12 M24 arm trained from scratch on the same
  frozen 20k-case ceiling.

## Frozen subset result

| Stage | Control cost | Candidate cost | Cost delta | Control V_rel | Candidate V_rel | V_rel delta |
|---|---:|---:|---:|---:|---:|---:|
| 2k / refiner400 | 2.61848 | 1.18017 | -54.93% | 0.19920 | 0.04020 | -79.82% |
| 5k / refiner1600 | 1.28123 | 1.10655 | -13.63% | 0.05471 | 0.03503 | -35.96% |

At confirmation, 28/32 cost cases and 21/32 V_rel cases improve; both paired
bootstrap intervals exclude zero; legalized feasibility is 1.0.

Component values at confirmation:

| Arm | HPWL gap | Area gap | b2b HPWL | p2b HPWL | Grouping | Boundary |
|---|---:|---:|---:|---:|---:|---:|
| depth12 scratch | 0.09025 | 0.17611 | 0.09906 | 0.94889 | 1.03125 | 1.84375 |
| Trinity-initialized | 0.00501 | 0.04065 | 0.00736 | -0.00528 | 0.53125 | 1.34375 |

## Claim boundary

This is frozen-subset confirmation evidence. It does not yet establish
Official-100, full-data, multiseed, external routed-PPA or CVPR SOTA status.
The final report must disclose that the backbone begins from the released anchor
and that the 5k fine-tuning run uses the frozen 20k-case ceiling.

The next gate is Official-100 at NFE4/refiner1600 under the same refiner and
legalizer protocol.

## Official-100 update

The same frozen checkpoint was evaluated on Official-100 at NFE4/refiner1600,
with group=2 and the common legalizer:

| Inference seed | Cost | V_rel | Feasible |
|---|---:|---:|---:|
| 19 | 1.07715 | 0.02175 | 1.0 |
| 20 | 1.08433 | 0.02358 | 1.0 |
| 21 | 1.09173 | 0.02397 | 1.0 |
| three-seed mean | 1.08440 | 0.02310 | 1.0 |

The seed19 matched Trinity anchor is `1.08803 / 0.02398`; the M27 seed mean is
therefore `-0.33%` cost and `-3.67%` V_rel against that anchor. This is a mean
win, not yet a case-paired significance claim: the seed19 paired intervals cross
zero. Exact per-seed paired anchor comparisons remain mandatory.

## Training-seed stability

Three additional M27 fine-tuning seeds were evaluated at fixed inference seed 19:

| Training seed | Cost | V_rel |
|---|---:|---:|
| 19 | 1.10689 | 0.02418 |
| 20 | 1.12421 | 0.02870 |
| 21 | 1.07530 | 0.01925 |
| mean | 1.10213 | 0.02404 |

The mean is `+1.30%` cost and `+0.24%` V_rel versus the seed19 group2 anchor; both
pooled intervals cross zero. Therefore the current full-backbone-LR recipe is
not training-seed stable and cannot support a seed-stable SOTA claim. The
architecture signal remains: seeds 21 and 23 beat the anchor on both means.

M28 tests an anchor-preserving optimizer recipe: typed encoder and hierarchy
heads use the normal LR, while the pretrained backbone uses 0.1x LR. The worst
observed training seed20 is evaluated first; it cannot be selected based on
Official-100 because the recipe and all-seed plan must be frozen in advance.

## Seed and recipe diagnosis

The following follow-ups were completed and retained as negatives:

| Probe | Official-100 cost | V_rel | Decision |
|---|---:|---:|---|
| M28 seed20, 0.1x backbone LR | 1.10186 | 0.02422 | close to anchor, still loses both |
| M29 full 50k, 20k finalist | 1.09334 | 0.02363 | mixed and non-significant |
| M30 frozen backbone adapters | 1.13970 | 0.02930 | rejected |
| M31 same-init seed soup | 1.10054 | 0.02538 | rejected |
| M32 seed20 fixed data order21 | 1.11008 | 0.02408 | rejected |
| M33 fixed typed init21, data19 | 1.15274 | 0.02808 | rejected |
| M33 fixed typed init21, data20 | 1.11115 | 0.02585 | rejected |

The search for stable training seed/order behavior is closed. Do not report the
best seed as a seed-stable SOTA method.

## Canonical mechanism candidate

The strongest reproducible recipe uses M27 training/data seed21:

- Official-100 cost `1.07530` versus anchor `1.08803` (`-1.17%`);
- V_rel `0.01925` versus `0.02398` (`-19.72%`);
- feasibility `1.0`;
- paired cost interval `[-0.06185, 0.03774]`;
- paired V_rel interval `[-0.01247, 0.00326]`.

This is a strong mean-level mechanism result, not a significance or seed-stable
SOTA result. Preserve the interval crossing and seed heterogeneity in every
paper-facing table.

## Canonical budget and component evidence

On the frozen 32-case screen, candidate NFE1/refiner400 is the best NFE point:
`1.16866 / 0.043995`. NFE2, NFE4 and NFE8 regress; this is not a monotonic
more-NFE-improves-quality curve.

At NFE4, refiner800 is the best candidate cost point (`1.16398 / 0.03197`).
Refiner1600 returns to `1.20420 / 0.03582`, so the 50-step quality-loss point
and the non-monotonic long-refiner behavior must remain visible.

An equal-native-budget Official-100 check compares candidate NFE1/refiner400
against Trinity NFE8/refiner400:

| Arm | Cost | V_rel | Feasible |
|---|---:|---:|---:|
| Candidate NFE1/ref400 | 1.21552 | 0.03707 | 1.0 |
| Trinity NFE8/ref400 | 1.22361 | 0.04380 | 1.0 |

The means improve by `-0.66%` cost and `-15.37%` V_rel, but both paired
intervals cross zero. This is an equal-budget mean result, not a significant
per-case win.

The initialization ablation is much stronger. Canonical M27 versus scratch
depth12 on Official-100 improves cost `10.36%`, V_rel `49.95%`, area gap
`68.66%`, b2b `88.47%`, p2b `77.98%`, grouping `50.67%`, boundary `46.73%` and
rho MAE `35.17%`. The paired intervals exclude zero for all except p2b.

Against the official anchor, the candidate improves mean cost, V_rel, b2b, p2b,
grouping, boundary and rho, but its area gap is `4.77%` worse and cost p90/p95
are slightly worse. Do not claim area/outline superiority over Trinity.

The attempted identity adapter did not reproduce Trinity's native sampling
source semantics and was not used as an anchor. Only native Official-100
Trinity runs are valid for anchor comparisons.
