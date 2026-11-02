# Chip Floorplanning Concepts in FloorFlow

Status date: 2026-10-27.

This note defines the chip-floorplanning vocabulary used by the FloorSet route in
this repository. It describes the data fields and formulas in the pinned Trinity
clone; it does not redefine FloorSet semantics by analogy with residential
floorplans.

For the end-to-end mapping from real physical-design actions to model pathways,
see `docs/chip_protocol_action_map_v2.md`.

Primary code references:

- `third_party/Trinity/src/trinity/floorplan/types.py`
- `third_party/Trinity/src/trinity/conditioning/features.py`
- `third_party/Trinity/src/trinity/floorplan/scoring/cost.py`
- `third_party/Trinity/src/trinity/floorplan/scoring/soft.py`
- `third_party/Trinity/src/trinity/floorplan/scoring/feasibility.py`
- `third_party/Trinity/docs/physics.md`
- `third_party/Trinity/docs/legalizer.md`
- FloorSet paper: <https://arxiv.org/abs/2405.05480>

## 1. The Placement Problem

A chip floorplan case contains rectangular partitions or blocks. The model must
assign each block a position and shape while respecting:

- target area and area tolerance;
- fixed-shape and preplaced constraints;
- a block-to-block connectivity graph;
- terminal/pin-to-block connections;
- grouping clusters;
- multi-instantiation groups;
- required outer-boundary edges.

The core objects are:

| Object | Current representation | Meaning |
|---|---|---|
| Block / partition | rectangle `(x, y, w, h)` | A macro, subsystem, or partition being placed |
| Case / instance | `FloorplanInstance` | Inputs: areas, constraints, netlist, pins, GT |
| Placement | `Placement` | One candidate set of block rectangles |
| Constraint row | `[fixed, preplaced, mib, cluster, boundary]` | Five native FloorSet constraint columns |
| Net | weighted edge | Connectivity used to calculate wirelength |
| Pin / terminal | `(x, y)` plus p2b edge | A fixed external connection point |

The model's latent is `(cx/s, cy/s, log(w/h))`, where
`s = sqrt(sum(block areas))`. Decoding preserves each block's target area exactly;
only center and aspect remain to be generated.

## 2. HPWL and Connectivity

HPWL means **half-perimeter wirelength**, using center-to-center Manhattan
distance under the FloorSet convention:

```text
HPWL =
  sum_(i,j in b2b) w_ij * (|cx_i-cx_j| + |cy_i-cy_j|)
  +
  sum_(p,b in p2b) w_pb * (|px_p-cx_b| + |py_p-cy_b|)
```

No square root or Euclidean distance is used. The two terms are added because
both represent physical connection length.

### b2b connectivity HPWL

`b2b` means **block-to-block**:

```text
(block_i, block_j, weight)
```

It comes from the chip netlist. `b2b connectivity` therefore means logical or
electrical connectivity, not necessarily geometric adjacency. Two blocks can be
far apart in the netlist graph even if their rectangles touch in the layout.

`b2b HPWL` measures how far connected block centers are from each other after
placement. Lower is better.

The project's `b2b_hpwl_gap` component is:

```text
(predicted_b2b_HPWL - GT_b2b_HPWL) / max(GT_b2b_HPWL, eps)
```

### p2b connectivity HPWL

`p2b` means **pin-to-block**:

```text
(pin_index, block_index, weight)
```

The pin position is fixed in the case data. The term measures the Manhattan
distance from that terminal/pin to the placed block's center. This is a
terminal/congestion-aware connection length, not the movement of internal
standard-cell pins.

The exact code term is `p2b`/pin-to-block. Calling it “port HPWL” is only a loose
analogy; `port` is not a separate stored field or metric in the current chip
pipeline.

## 3. Area and Outline

The latent decoder preserves each block's area. Therefore `area_gap` is not
per-block area error.

The official gap is:

```text
area_gap =
  (predicted_layout_bbox_area - target_layout_bbox_area)
  / max(target_layout_bbox_area, eps)
```

It measures how large the bounding rectangle of the whole placement is relative
to the reference layout. In papers and experiment notes, we call this
**area/outline gap** or **compactness gap**.

It is not:

- sum of block-area errors;
- mismatch of two geometric outline polygons;
- fixed-outline violation.

FloorSet official cases do not carry a fixed outline in this pipeline. The
fixed-outline `outline` hinge exists only for the GSRC/MCNC bookshelf protocol.

## 4. Soft Constraints and V_rel

The official hard checker first requires:

- no overlap;
- soft blocks within 1% of target area;
- fixed blocks preserve shape;
- preplaced blocks preserve position and shape.

If these fail, the hard cost is `10.0`.

For feasible layouts, the score includes three **counted** soft constraints:

```text
V_grouping = sum_clusters (connected_components(cluster) - 1)
V_mib      = sum_MIB_groups (distinct_shapes(group) - 1)
V_boundary = count(boundary-coded blocks missing a required outer edge)

V_rel = (V_grouping + V_mib + V_boundary) / N_soft
```

`N_soft` is the number of available violation slots:

```text
N_soft =
  boundary-coded blocks
  + sum_clusters(|cluster|-1)
  + sum_MIB_groups(|group|-1)
```

Thus `V_rel` is a normalized discrete soft-violation ratio in `[0, 1]`.
It does **not** contain HPWL, area gap, opening error, or per-block area error.

### Grouping / cluster V_rel

A cluster is a set of blocks that should remain physically grouped or abutted,
for example blocks in the same power domain. In FloorSet, grouping is an
edge-connected-component constraint.

`grouping V_rel` is the count-based normalized violation:

```text
grouping_vrel = V_grouping / N_soft
```

It is not the continuous grouping gap used during training. Training uses an MST
gap-to-contact penalty; the official `V_rel` uses a discrete component count.

### MIB

`MIB` is the code field name for FloorSet's
**multi-instantiation constraint**. The FloorSet paper describes it as multiple
partitions that are instances of the same master partition, for example eight
copies of a CPU core. They must share the same `(w, h)` shape.

Members already share target area, so one shared shape means one shared
`rho = log(w/h)`. The official violation is:

```text
V_mib_for_group = number_of_distinct_(w,h)_shapes - 1
```

This is shape-sharing across instances. It is unrelated to area tolerance,
boundary edges, or preplacement.

The repository preserves the short field name `MIB`; the important operational
definition is “multi-instance blocks sharing one shape.”

### Boundary V_rel

`boundary_code` specifies required edges of the placement's global bounding box:

- bit `1`: left;
- bit `2`: right;
- bit `4`: top;
- bit `8`: bottom;
- corners are bit sums.

A coded block must touch each required bounding-box edge within `1e-6`.
`boundary V_rel` normalizes the count of coded blocks missing any required edge.

This is an **outer floorplan edge-alignment constraint**. It is not:

- the boundary between two adjacent modules;
- an internal wall;
- a fixed-outline size violation;
- an opening.

## 5. Opening Is a Different Domain

In the repository's residential/CubiCasa floorplan branch, an **opening** means a
door, window, or other passage in a wall.

The chip/FloorSet branch has no opening channel and no opening metric. A
“reserved region” is also not represented by the current FloorSet fields.

Therefore:

- do not call `boundary V_rel` an opening metric;
- do not call `p2b` an opening metric;
- do not claim opening-specific improvements for the chip route.

## 6. Raw, Refined, and Legalized Scores

The pipeline stages have different meanings:

| Stage | Meaning |
|---|---|
| Raw flow / NFE | Generated blocks before a task-specific refiner or legalizer |
| Closed-form refinement | Differentiable soft-constraint descent on the generated latent |
| Legalization | `scale_pack`: restores exact area/shape and packs blocks into a legal layout |
| Hard score | FloorSet checker after legalization |

All official hard costs reported in the chip experiments are on legalized
layouts. Raw feasible fraction can be zero even when legalized feasible fraction
is one.

The official feasible hard cost is:

```text
cost = (1 + 0.5 * (max(hpwl_gap,0) + max(area_gap,0)))
       * exp(2 * V_rel)
```

with cost `10.0` if infeasible.

## 7. Common Confusions

| Confusion | Correct distinction |
|---|---|
| b2b adjacency = geometric adjacency | b2b is netlist connectivity |
| p2b = port placement | p2b is pin/terminal-to-block wirelength |
| area_gap = block area error | block areas are exact; area_gap is bbox compactness |
| grouping = graph community | grouping is a native cluster constraint |
| MIB = regular block | MIB groups require shared shape across instances |
| boundary = module boundary | boundary is outer bbox edge alignment |
| opening exists in chip route | openings occur only in the residential branch |
| V_rel includes area/HPWL | V_rel includes grouping + MIB + boundary counts only |

## 8. Claim-Safe Language

Supported:

- “Graph-token flow reduces b2b net HPWL.”
- “The improvement is concentrated in area/outline and grouping.”
- “Cost-tail reduction indicates fewer structural failures.”
- “MIB has no official-100 signal in the current comparison.”

Unsupported:

- “The method improves openings.”
- “The method improves MIB.”
- “The method solves the highest-b2b or highest-pin strata.”
- “The area gap is per-block area error.”
- “The component gap proves that the exact-area decoder causes the Trinity
  difference.”
