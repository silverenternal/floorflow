# Chip Legalizer Research Update v1

Date: 2026-11-01

## Verified Sources

Primary source: <https://arxiv.org/abs/2608.24946>

MacroAgent frames macro legalization as clustering, contour generation,
template matching and inter-cluster refinement. Its reported gains are
regularity first, followed by `3%-5%` routed-wirelength reduction with
comparable congestion. This is directly aligned with the observed PlanB
failure: additive group residuals can improve a raw proxy while the global
packer destroys the structural gain.

Primary source: <https://arxiv.org/abs/2610.06489>

GPlaceRL represents legalized placements as graphs and studies bounded local
placement actions. This supports extending the current deterministic
component-move repair into a learned target-free action policy after the
deterministic contract is certified.

Primary source: <https://arxiv.org/abs/2512.01242>

The tangram/constraint study reports that continuous diffusion struggles when
feasible solutions occupy small or disconnected low-dimensional sets. This
matches the S1 result and argues for local feasible-space moves rather than
another unconstrained velocity residual.

Primary source: <https://arxiv.org/abs/2604.23658>

FlowPlace uses hard-constraint sampling for overlap-free layouts. PlanB keeps
the pinned global legalizer and adds a separately verified post-legalizer
repair, so the baseline protocol is not silently changed.

## Consequence for PlanB

The next model-level repair should learn which legalized cluster component to
move and where, using the already proven target-free action space. The action
policy must preserve hard feasibility, boundary count and anchored blocks, and
must be ablating against the deterministic repair and a no-action control.

