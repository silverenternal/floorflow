# S1 Group-Moment State v1

## Purpose

The existing L1 state stores one three-value latent mean per cluster, MIB or
boundary slot. That cannot represent group span, aspect, area share, internal
spacing or disconnected components, which explains why dynamic graph input
improved WL/area/boundary while collapsing grouping.

## Implementation

`GroupMomentCoupler` reuses the seven target-free geometry features already
produced by `StructuralRelationHead`:

1. group center x/y;
2. group width/height;
3. group area fraction;
4. mean intra-group gap;
5. component deficit.

The existing structural-relation head predicts these moments. The new coupler
aggregates each block's active group moments and injects a three-value
center/rho correction into the per-block trajectory. It is opt-in through
`group_moment_conditioning=True`. The residual head starts at zero while its
outer gate starts at one, so enabling the field preserves existing M27 outputs
exactly before training without creating a multiplicative zero-gradient gate.

`configs/chip_s1_group_moment_v1.json` defines the first frozen-subset screen.
The first training scope is `b2b+downstream`; no GPU run is started by the CPU
contract phase.

## Gate

The candidate must preserve exact area and feasibility, avoid grouping, b2b and
area regression, improve at least two structural components, and raise
constraint-safe joint wins by at least five percentage points on the 500-case
dev screen. A 32-case movement alone is insufficient.

## Typed Semantic Revision v2

The first pooled moment residual passed the CPU contracts but failed the
500-case expansion: grouping regressed 6.08% and V_rel 1.33%. V2 keeps the
public hierarchy state unchanged but separates moment codes into four cluster
slots, one MIB slot and four boundary slots before pooling. It also applies a
smooth early-explore/late-commit schedule to the group-to-block correction and
retains the zero gate.

The v2 candidate is defined by `configs/chip_s1_typed_group_moment_v2.json`.
The v1 result remains negative evidence; no v1 checkpoint is promoted.

## Native Moment-State Revision v3

V2 also failed its matched 32-case probe: grouping regressed 0.45% and total
regressed 0.024% despite area, boundary and overlap improvements. V3 moves the
seven moments into the optional hierarchy state itself:

- target construction creates supervised moment targets;
- source construction creates a target-free moment state;
- interpolation, hierarchy loss, ODE advancement and final projection preserve it;
- a zero-initialized native head predicts the next moment state;
- block conditioning consumes this flowing state rather than a post-hoc residual.

The default M27 path keeps `structural_moment_state=None` and remains an exact
identity. Configuration: `configs/chip_s1_native_group_moment_v3.json`.

## Target Semantics Repair v4

The three rejected variants shared a target-definition error. Group "width" and
"height" were computed from the mean of each member's left/right or top/bottom
edges. That measures mean member size, not the occupied group bbox: two
one-unit members at `x=0` and `x=10` produced span `1` instead of `11`.

V4 computes each active slot's center and extent from masked membership `amin`
and `amax`. The representation is now sensitive to group relocation, separation
and scale while retaining the same seven-feature contract, zero-mask behavior
and disabled-path identity. This is a target-semantics repair; a new checkpoint
must still pass the predeclared quick screen before any promotion.

## Verification

- targeted hierarchy/group-moment and true group-span tests: pass;
- architecture capability audit: pass;
- first-party Python line limit: every file below 600 lines.
