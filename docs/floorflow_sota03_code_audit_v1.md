# FloorFlow SOTA-03 Code Audit

## Scope

This audit covers the FloorFlow residential path only. It does not inspect or
modify the chips/FloorSet implementation or the active OpenROAD process. No
test503 case is used, and the audit does not turn a diagnostic number into an
external SOTA claim.

## What Already Exists

- `CanonicalFloorplan` is the shared room/wall/opening/furniture object used by
  the residential protocol evaluator.
- `CadActionGraph` encodes the production order `envelope -> program -> rooms ->
  walls -> junctions -> openings -> furniture -> validation`.
- The predicted-control wall scorer already has candidate-graph attention,
  exterior/interior budgets, a filtration descriptor, and typed wall outputs.
- `RasterWallPlan` remains the mature vector readout used by the wall, opening,
  room, and furniture stages.
- `partition_polygon_rows` already produces exact room-cell polygons from dense
  partitions.

## Material Gaps Found

| Area | Finding | Resolution |
| --- | --- | --- |
| Canonical room schema | Room holes were accepted by downstream readouts but absent from the canonical object. | Added first-class hole rings and render support. |
| CAD graph | Junctions and representation-changing operations had no canonical fields. | Added typed junctions and explicit operations. |
| Determinism | Room order, polygon winding, ring rotation, and wall direction depended on input order. | Added canonical room/wall ordering, exterior winding, hole winding, and stable ring rotation. |
| Raster authority | The evaluator could rasterize overlapping rooms by overwrite order. | Vector overlap now fails explicitly; raster boundary collisions are reported separately. |
| Representation bridge | Existing `RasterWallPlan` outputs had no route into the unified token object. | Added `canonical_plan_from_readout` without re-rasterizing or moving vertices. |
| Topology descriptor | The four filtration thresholds reused one threshold-independent edge count. | Induced edges are now counted only when both endpoint nodes pass the current threshold. |
| Token contract | No reversible vector grammar existed for the canonical CAD object. | Added a fixed-point, typed grammar with explicit delimiters, ring boundaries, and `END`. |

## Prior Topology Paper Integration

The referenced NeurIPS 2026 work is *A Topological Encoder Decoder Framework for
Temporal Graph Learning* (`third_party/TopoGED`, commit
`fde9441dc666c0dac5036f8d88cd46c190ad6e49`). It is a temporal graph model,
not a floorplan model. A direct import of its temporal predictor would be
out-of-domain.

The defensible FloorFlow integration is narrower:

1. multiscale degree-filtration descriptors over the wall-candidate graph;
2. global and phase-specific reconstruction budgets;
3. shared-junction candidate attention;
4. ordered multi-phase decoding.

These mechanisms already map to FloorFlow's exterior/interior wall phases and
CadActionGraph. The audit fixed a real semantic defect in item 1 rather than
adding another out-of-domain network.

## Unified Token Contract

The grammar is implemented in `floorflow-unified-token-v1`:

```text
FOOTPRINT ring
OPERATION*
ROOM exterior hole* support
ADJACENCY*
WALL*
JUNCTION*
OPENING*
FURNITURE polygon
END
```

Coordinates use a declared fixed-point scale. The default is 1/16 pixel, while
the current synthetic CubiCasa-cell audit uses integer coordinates and therefore
reports exactly zero quantization error. Every ring has its own begin and end
token. The decoder rejects partial sequences, trailing tokens, duplicate IDs,
unknown references, overlapping rooms, and detached openings.

Vector geometry is canonical. Raster maps are diagnostics rendered from the
decoded vectors. Shared room boundaries can collide on the same raster pixel,
but this is reported separately from positive-area polygon overlap.

## Audit Results

`scripts/audit_unified_floorplan_tokens.py` writes
`outputs/reports/unified_floorplan_token_audit_v1.json`.

| Gate | Result |
| --- | ---: |
| Exact canonical round-trip | 1.000 |
| Exact token re-encode | 1.000 |
| Input-order and winding invariance | 1.000 |
| Explicit end token | 1.000 |
| Malformed-input rejection | 1.000 |
| Positive-area room overlap | 0 |
| Outside-footprint room pixels | 0 |

The two audit cases use 61 tokens for the base two-room plan and 68 tokens when
a hole and operation are added. Token bytes are 0.715 and 0.737 of the canonical
JSON bytes respectively. Median encode/decode latency is about 1.3-1.4 ms on CPU.
These latency values are contract-audit measurements, not model inference
throughput claims.

### Corrected Filtration Mechanism Screen

The corrected induced-edge descriptor was re-run with the existing matched
64-train/32-validation, eight-epoch wall-candidate protocol on `cuda:0`.

| Metric | Evidence control | Corrected descriptor | Delta |
| --- | ---: | ---: | ---: |
| Eligible Wall F1 | 0.353893 | 0.359744 | +0.005851 |
| All-case Wall F1 | 0.352020 | 0.354861 | +0.002842 |
| Eligible Opening F1 | 0.086830 | 0.086830 | 0.000000 |
| Eligible Room F1 | 0.226042 | 0.226042 | 0.000000 |
| Eligible validity | 1.000 | 1.000 | 0.000000 |
| Eligible watertightness | 1.000 | 1.000 | 0.000000 |

The final three checkpoints are all above the matched control and improve
monotonically. This is a positive mechanism signal after fixing the descriptor,
not a promotion result or an external SOTA claim. The summary is stored in
`outputs/reports/topoged_induced_filtration_fast64_audit_v1.json`.

## Engineering Checks

- Full test suite: 972 passed.
- Scoped Ruff checks: clean.
- No Python file under `src/`, `scripts/`, or `tests/` exceeds 600 lines.
- The token audit report and `fptodo.json` parse as valid JSON.
- No FloorFlow training process was present before the audit.
- The active chips process was not modified.
- The CAD action/layer contract audit passed all 18 checks on `cuda:0`; its
  report is `outputs/reports/cad_action_layer_contract_audit_v3.json`.

Checkpoint loading still uses `weights_only=False` broadly because current
checkpoints contain taxonomy/config objects. Treat downloaded checkpoints as
trusted artifacts or add a signed allowlist before evaluating third-party files.
Most training entry points also seed RNGs but do not request fully deterministic
CUDA kernels; replicate-seed differences must remain visible in confidence
intervals rather than being described as bitwise reproducibility.

## Next Code Priorities

1. Re-screen the corrected induced-edge filtration loss on the matched 64/32
   mechanism split.
2. Promote only the unified generator that consumes the token/CAD schema; do not
   add another raster-only endpoint.
3. Map each CadActionGraph stage to a typed emitted object and an explicit
   downstream consumer before adding capacity.
4. Keep external Raster2Seq geometry behind the external-baseline boundary until
   a one-to-one matched split is executable.
