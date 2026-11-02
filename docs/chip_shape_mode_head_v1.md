# Chip Shape-Mode Head v1

Date: 2026-11-01

## Purpose

Existing rho reranking and median-shape templates cannot expand the candidate
pool. This opt-in head produces several rho hypotheses from one shared
endpoint, without another backbone pass or adding the same scalar residual to
every candidate.

## Architecture

`ShapeModeHead` consumes:

- per-block hidden tokens;
- per-block center/rho state;
- pooled cluster/MIB/boundary context through the existing membership;
- the typed global token;
- one learnable embedding per shape mode.

The shared trunk predicts bounded rho offsets. Centers and exact block areas
are unchanged. The interface returns:

- `shape_mode_rho_residuals [B,K,N]`;
- `shape_mode_latents [B,K,N,3]`.

Before candidate export, fixed/preplaced residuals are zeroed and MIB members
share one median rho hypothesis. `sample_hierarchy_flow` returns the K latents
in its diagnostics when the head is enabled. This keeps candidate construction
inside the same target-free graph/constraint contract as the base model.

`attach_shape_mode_head` adds this result through a forward hook on the existing
`HierarchicalTokenFlowModel`, so the 599-line core model does not grow and all
existing `isinstance` train/eval paths remain valid.

## Identity and Training Contract

- final rho projection is zero-initialized;
- outer gate starts at one;
- all modes return exact M27 identity at initialization;
- head gradients are non-zero at identity;
- default configuration has zero shape modes.

Training supports:

- winner-takes-all smooth-L1 over full mode latents;
- pairwise rho diversity below a declared margin;
- soft mode-usage entropy balancing.

The auxiliary loss is opt-in through `shape_mode_loss_weight`.

## Next GPU Screen

Run one 300-step probe on the frozen 1k subset with four modes and weight `0.1`.
Promote to 2k only if internal dev rho improves and total/group/overlap remain
non-regressing. Candidate value must then be measured after the pinned
exact-area projector and legalizer on the frozen 92/500 protocol.

