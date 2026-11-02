# Chip Operator Optimization v46

## Scope

Optimization must strengthen the current chip architecture rather than replace it:

- sparse typed graph memory;
- explicit L0-L3 hierarchy states;
- per-block flow trajectory;
- exact-area decoding;
- hard-constraint projection and closed-form refiner.

## Initial production profile

Source: `outputs/reports/chip_v46_operator_profile_smoke.json`.

- Device: `cuda:1` fallback because `cuda:0` was occupied.
- Block count: 40.
- Batch size: 16.
- Typed context: `5.89 ms`.
- Training step without projector/refiner: `32.65 ms`.
- NFE2 sampling with projector: `92.10 ms`.
- One eight-iteration projector call: `31.56 ms`.

The main GPU operators include matrix multiplication, linear/autograd, RMS norm,
sum, attention backward and indexing. CPU-side launch overhead is also large.
The projector is not a minor post-step at this scale: one call costs roughly as
much as the measured typed training step.

## Accepted operator candidate

The fixed/preplaced/MIB projection masks and anchored target medians are invariant
across the eight iterations of `project_hard_constraints`. They are now prepared
once and reused.

Paired benchmarks with identical diagnostics:

| Stratum | Legacy | Prepared plan | Speedup |
|---|---:|---:|---:|
| 40 blocks, batch 16 | 40.58 ms | 35.77 ms | 1.134x |
| 60 blocks, batch 32 | 34.25 ms | 29.57 ms | 1.158x |

Forward output is exactly equal on both strata (`max_abs_difference = 0`). This
candidate still requires the full legalized-dev smoke and timing guard before it
is treated as an accepted optimization.

A deterministic five-case legalized regression was run twice on `cuda:1`. All
reported primary and structural metrics matched exactly between the two current
runs. Repeating the same five cases on `cuda:0` reproduces every historical
pre-change metric exactly. A cross-device `cuda:1` run differs on one grouping
case, isolating that discrepancy as device nondeterminism rather than an operator
change.

`torch.compile(..., mode="reduce-overhead")` on the projector path produced
`3.84 ms` versus `26.93 ms` for the prepared eager path (`7.01x`) at block40 /
batch16. Its maximum output difference from the prepared path was `4.53e-6`.
The latent-only compiled path reached `0.76 ms` (`34.85x` faster than prepared
eager), but this speed cannot be adopted by itself. Through a four-step ODE, the
same small one-call difference amplified to `1.127` maximum latent difference,
changing the generated trajectory materially. The compiled projector is therefore
rejected as a default path and remains available only for explicit quality
experiments. It intentionally skips per-step scalar diagnostics; any experiment
using it must recompute certification on the selected final layout.

The duplicated hierarchy/typed masked mean was replaced by one shared
`masked_token_mean` implementation using batched matrix multiplication.
Microbenchmarks at batch16/hidden192 gave `1.127x` for 40 block tokens,
`1.155x` for 9 structural tokens and `1.255x` for 8 port tokens. Forward outputs
and gradients were bit-exact in all three shapes.
The combined projection-plan + masked-reduction implementation also reproduced
all five legalized regression metrics exactly on `cuda:0`, and the complete
32-case screen again matched every historical metric exactly.

## Optimization order

1. Profile projector internals and exact-area decode separately.
2. Fuse typed masked reductions and repeated incidence/group reductions.
3. Compare vectorized overlap separation with current projector loops while
   preserving exact fixed-shape and area semantics.
4. Test `torch.compile` on static backbone and projector subgraphs.
5. Use MiTA FlashAttention or DHSA block-sparse kernels only when typed-memory
   length and mask occupancy make the specialized dispatch viable.
6. Validate numerical equivalence and legality before reporting latency.

Any saved time is returned to matched-compute experiments; an operator speedup is
not itself a model-quality result.
