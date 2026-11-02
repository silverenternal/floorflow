# FEVP v1.0-Hardened Formal Specification

This appendix-level specification defines the executable protocol implemented by
`floorflow.evaluation.fevp_metric_contracts`.

## 1. Notation

- `L_before, L_after in {0,...,K-1}^{H x W}`: canonical `uint8` label maps.
- `P`: footprint support.
- `R`: declared edit region.
- `C`: target-change mask.
- `W`: wall support in either before or after target.
- `D1(W)`: one-pixel dilation.
- Valid outside domain:

```text
O = P and not R and not D1(W)
```

- Pixel tolerance is exactly zero.

## 2. Pixel Outside Exact

```text
outside_pixel_exact =
    1[ L_after[x] = L_before[x] for every x in O ]
```

The implementation computes the fraction over `O`; a value of 1 means every
outside pixel is an exact `uint8` match.

## 3. Room Instances and Matching

For each legal room class, connected components produce room instances
`I_before`, `I_after` with masks, semantic labels, and polygons.

Matching is one-to-one and same-label:

```text
IoU(i,j) = |M_i intersect M_j| / |M_i union M_j|
m(i) = argmax_j IoU(i,j), constrained to one-to-one assignment
```

An assignment is accepted iff `IoU >= 0.5`. The assignment is solved by the
Hungarian algorithm.

## 4. Three-Layer Outside Exact

Let `U` be before-edit room instances with:

```text
U = { i : |M_i intersect R| = 0 and |M_i intersect C| = 0 }
```

### 4.1 Room label exact

```text
outside_room_label_exact = 1[ every i in U is matched and has the same label ]
```

### 4.2 Room mask exact

```text
outside_room_mask_exact = 1[ m(M_i) = M_i for every i in U ]
```

### 4.3 Polygon exact

```text
outside_room_polygon_exact =
    1[ Hausdorff(boundary_i, boundary_m(i)) <= 0.5 px for every i in U ]
```

### 4.4 Edge exact

An untouched-room edge exists when two room polygon boundaries are at most
two pixels apart. Edges are compared after one-to-one room mapping:

```text
outside_edge_exact = 1[ E_before(U) = E_after(m(U)) ]
```

### 4.5 Joint room-level exact

```text
outside_room_level_exact =
    outside_room_label_exact
    and outside_room_mask_exact
    and outside_room_polygon_exact
    and outside_edge_exact
```

### 4.6 Locality Violation Rate

```text
VLR = |{x in O : L_after[x] != L_before[x]}| / |O|
```

## 5. Five-Check Validity

```text
V = V_outer and V_room_ring and V_self_intersection
    and V_no_overlap and V_semantic
```

Definitions:

- `V_outer`: footprint has a non-degenerate closed exterior contour.
- `V_room_ring`: every room polygon has at least three vertices, positive area,
  and a closed ring.
- `V_self_intersection`: every raw room polygon satisfies `Polygon.is_valid`.
- `V_no_overlap`: pairwise room intersection area is at most `4 px2`.
- `V_semantic`: every declared room has a legal taxonomy label.

The report must include all five booleans and all failure codes.

## 6. Conditional Watertightness

A case is conditionally eligible when its target is watertight:

```text
T_cond = T_prediction evaluated only on {target_watertight = true}
```

The tile test is:

```text
room polygons are valid
and pairwise overlap <= 4 px2
and union has one connected component
and no interior hole area > 0.25 px2
```

Hole diagnostics fill the raster room-support mask and count enclosed unlabelled
components above `0.25 px2`. Wall endpoints clustered within two pixels with
degree one are dangling endpoints.

## 7. Node and Edge Metrics

Let node sets be unique semantic room types and edge sets be canonical type
pairs.

```text
node_precision = |N_pred intersect N_gt| / |N_pred|
node_recall    = |N_pred intersect N_gt| / |N_gt|
node_f1        = 2PR / (P + R)

edge_precision = |E_pred intersect E_gt| / |E_pred|
edge_recall    = |E_pred intersect E_gt| / |E_gt|
edge_f1        = 2PR / (P + R)
edge_overlap   = |E_pred intersect E_gt| / |E_pred union E_gt|
```

## 8. Fixed-Evaluator GED

NetworkX attributed GED uses:

- node substitution cost `0` for equal room type, otherwise `1`;
- node insertion/deletion cost `1`;
- edge insertion/deletion cost `1`;
- edge substitution cost `0`;
- timeout `0.5 s`; timeouts are recorded.

GED claims use only paired differences under this evaluator.

## 9. Delta Metrics

For valid edit region `V_R = R and valid`:

```text
expected_change = (target_before != target_after) and V_R
predicted_change = (prediction_before != prediction_after) and V_R
Delta-IoU = |expected_change intersect predicted_change|
            / |expected_change union predicted_change|
Delta-MSE = mean((expected_change - predicted_change)^2 over V_R)
```

## 10. Protocol Attack Predicate

An evaluator is sensitive for metric `m` when an attack changes only the intended
semantic corruption and:

```text
attack_sensitive(m) =
    m_attacked < m_baseline for maximize metrics
    or
    m_attacked > m_baseline for minimize metrics
```

`false_pass = not attack_sensitive`. Protocol attacks test evaluator sensitivity;
they do not increase model scores.

The outside-exactness attack set includes:

- Move one untouched room mask by 1 px.
- Change one untouched room label.
- Change one untouched adjacency relation without moving pixels.

Each attack is applied to an otherwise identical case. A zero observed
false-pass rate is required before the protocol is considered hardened.
