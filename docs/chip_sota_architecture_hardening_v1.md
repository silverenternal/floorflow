# Chip SOTA Architecture Hardening v1

Status date: 2026-11-01. This is a CPU implementation package. It does not run
GPU training, ORFS or a long experiment, and it does not establish SOTA.

## Execution order

1. make hierarchy state causally affect block coordinates;
2. couple shape, pin tails and MIB sharing;
3. align deployment source and short-NFE endpoint behavior;
4. make legalizer reshape/trust explicit and reproducible;
5. rank checkpoints by joint legalized metrics;
6. turn M75 portfolio search into a trainable consensus target;
7. verify the complete global/semantic/hyperedge integration;
8. run terminal project audits.

## Coordinate reparameterization

`hierarchy_coordinates.py` is an opt-in, parameter-free transform. Active cluster
members use the predicted L1 cluster center and a bounded cluster scale derived
from the current layout. Unassigned blocks use the predicted L0 envelope.

The backbone therefore receives local centers rather than global centers, and
its prediction is mapped back before output adapters. The transform is
reversible, translation-equivariant within a cluster and keeps the pretrained
path unchanged when disabled.

`hierarchy_coordinate_mode=envelope` isolates the L0-only control;
`envelope_cluster` activates L1 cluster frames.

The enabled branch starts with a zero-initialized learnable coordinate blend, so
old checkpoints recover the exact global-coordinate path at initialization.
Cluster scales are bounded to a stable fraction of the envelope.

The gate is computed per block from the typed block token rather than shared as
one scalar. All coordinate-gate parameters are excluded from weight decay.

## Raw local-geometry encoder

The legacy 19 columns remain unchanged. A separate zero-gated learned encoder
adds target-free statistics that the handcrafted schema previously compressed:
weighted/max b2b degree, neighborhood fraction, weighted/max p2b demand, pin
centroid, pin spread and normalized pin strength. It starts as an exact no-op and
can learn locally important edge distribution without leaking target geometry.

This replaces the failed idea of repeatedly adding stronger hierarchy residuals
to an unchanged global coordinate space.

The gate is the only zero-initialized component. The internal projection keeps a
normal initialization so the first backward pass reaches the gate; zeroing both
the gate and output projection would make the branch permanently gradient-dead.
The same repair applies to global-spec, semantic, netlist-hyperedge and
structural-token residual gates.

## Branch health and objective balance

`branch_health.py` records parameter count, gate presence, maximum gate magnitude,
maximum gradient norm and finiteness for every opt-in branch. A gated branch must
have both an open gate and a reachable gradient; an ungated branch is judged by
gradient reachability. `scripts/train_chip_slot.py` writes this report at each
diagnostic checkpoint.

`multi_objective_balance.py` provides an opt-in inverse-scale objective balance.
Scales are measured under `no_grad`, normalized to mean weight one, and applied to
the original tensors so every objective retains its gradient path. Fixed weights
remain the default.

`branch_registry.py` is the canonical source for branch scope and stage. The
mainline recipe is `configs/chip_architecture_mainline_v4.json`; internal FloorSet
runs reject external-only global-spec and semantic conditioning.

## Joint shape, pin and MIB endpoint terms

`shape_pin_consistency.py` supplies reusable endpoint terms for:

- squared shared-rho deviation inside MIB groups;
- top-fraction pin-to-block distance;
- rho change outside a declared trust radius.

The terms are composed into the existing endpoint compatibility path rather
than creating a second training loop.

## Deployment source and NFE consistency

New recipes can require non-random block and level sources. The optional
`nfe_consistency` objective runs the current hierarchy for up to a configured
number of additional short Euler steps and supervises both the resulting
endpoint and its agreement with the first prediction.

The objective is disabled by default and is not a replacement for the
preregistered primary endpoint.

## Legalizer policy

`LegalizerPolicy` centralizes relation, reshape, trust, outline and timeout
settings. The official evaluator and external candidate legalization can now
run the same policy. `chip_legalizer_ablation_v1.json` freezes default,
shape-preserving, low-trust and expanded-relation comparisons.

This directly addresses the observed raw-to-legalized metric reversal and the
63%-81% reshape rates seen in M48.

## Checkpoint selection

Feasibility remains the first guard. Within feasible candidates, the existing
shortlist ranking now compares legalized cost and V_rel before area, b2b,
grouping, boundary, displacement, solver rung and raw validation total.

## Candidate consensus distillation

`candidate_consensus_training.py` defines a sharded NPZ contract and an
objective combining:

- fused endpoint supervision;
- lower-is-better candidate teacher ranking;
- selection entropy regularization;
- masked invalid candidates and a required reference candidate.

`scripts/train_chip_candidate_consensus.py` is a CPU runner. It does not reopen
the closed 32-case M76 selection cohort; a larger independent cohort is required.
`scripts/build_chip_candidate_consensus_cohort.py` converts a frozen candidate
manifest into one fixed-shape NPZ shard per case.
`scripts/export_chip_candidate_consensus_manifest.py` merges matched case
artifacts that contain raw or legalized xywh into that manifest.
The exporter requires one declared reference source, sorts every other source,
and rejects geometry/netlist input mismatches across checkpoints.

## Conditioning completion

Global design specification, node/port semantics, dynamic netlist hyperedges
and coordinate frames are exercised together in the external-graph to hierarchy
forward contract. Dynamic hyperedges now use net-port incidence rather than
discarding it after validation.

Dynamic port state and port-token trajectories are pooled per net and mixed into
the block hyperedge message. The block branch remains zero-gated.

## Legalizer and data controls

`scripts/run_chip_legalizer_counterfactual.py` replays every legalizer policy on
identical refined boxes. The official evaluator can retain raw xywh fields for
this purpose.

The progressive complexity sampler starts on low block-count strata and expands
to higher bands within a fixed training horizon. It remains opt-in and preserves
balanced b2b/pin-density groups inside each active band.

## Official-cost alignment

`official_cost_surrogate.py` mirrors the official structure
`(1 + 0.5*(HPWL gap + area gap)) * exp(2*soft_V_rel)`. It uses exact target
geometry only during training, clamps gaps and uses a smooth V_rel proxy built
from grouping, MIB and boundary evidence. The surrogate is opt-in and separately
ablatable from the legacy physics objective.
The objective supports delayed and annealed introduction so the mean layout
learns before the exponential tail dominates.

## Configuration

- `configs/chip_sota_architecture_hardening_v1.json`
- `configs/chip_legalizer_ablation_v1.json`
- `configs/chip_candidate_consensus_v1.json`
- `configs/chip_sota_staged_experiment_plan_v2.json`

The all-on hardening file is an integration contract, not the first training
arm. The staged plan separates coordinate warm-start, cluster frames,
source/NFE, shape constraints, graph semantics and consensus distillation.

## Evidence boundary

Passing tests proves implementation, shape, masking, checkpoint and gradient
contracts. It does not prove lower cost, lower V_rel, better area/b2b,
routed-PPA transfer or SOTA.
