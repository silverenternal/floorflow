# Chip Flow Source Contract v1

Date: 2026-10-31

Scope: hierarchical chip flow only. The numbers below are frozen-subset
mechanism-screen results, not full-data, routed-PPA or SOTA evidence.

## Finding

The typed-hierarchy training path and deployment path did not use the same block
source:

- training: independent Gaussian `block_state`;
- deployment: deterministic `source_latent` produced by `shelf_pack`.

The older non-hierarchical trainer already used the deployed shelf source plus
Gaussian noise. The hierarchy migration dropped that contract.

This mismatch creates a trajectory-consistency problem. The learned endpoint
field is supervised on a path the sampler does not execute. It can favor low-NFE
behavior while leaving a higher-refiner-budget ceiling, which matches the
retained matched-compute reversal.

## Implementation

`hierarchical_flow_runtime.hierarchical_training_state` now supports:

- `random`: retained for exact historical reproduction;
- `shelf`: trains from the deployed block source;
- `noisy_shelf`: shelf source plus predeclared Gaussian perturbation
  (one M22 point only, not a sweep);
- `source_levels=derived`: derives L0 bbox, L1 cluster/MIB/boundary and L3 port
  source states from the shelf geometry, without reading `target_latent`.

Checkpoint configs restore `hierarchy_source_levels`, so evaluation uses the
same source contract as training. The mechanism adds no model parameters.

## Frozen quick screen

Matched control: M20 random-source warm start. All arms use the frozen 5k
training subset, 2k steps and NFE4/refiner400 on the frozen 32-case screen.

| Mechanism | Cost | V_rel | Grouping | Boundary | b2b HPWL | p2b HPWL | Area |
|---|---:|---:|---:|---:|---:|---:|---:|
| M20 shelf block only | -1.10% | +10.28% | +7.32% | +6.00% | +5.01% | -88.69% | +0.43% |
| M21 shelf + derived levels | -1.06% | +1.21% | -7.32% | +2.00% | +5.19% | -67.99% | +9.30% |
| M22 noise 0.1 + derived levels | -3.97% | -0.84% | -9.76% | +6.00% | -13.73% | -68.92% | -13.22% |

M22 passes both quick primary metrics with feasibility 1.0. Its gain is
tail-driven: the lower half of the control cost distribution worsens, while the
p90/p95 tails improve. Confirmation at 5k/refiner1600 is mandatory before any
candidate registration.

## Boundary

- Do not report these numbers as full-data or external PPA results.
- Do not sweep source noise after observing M22.
- Do not call the source repair an architecture SOTA claim by itself.
- If confirmation fails, retain the train/deploy mismatch and the differential
  tail signal as mechanism evidence.
