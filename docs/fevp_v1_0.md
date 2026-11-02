# FloorPlan Editing Verification Protocol v1.0

Status: executable experiment protocol, not a manuscript.

## Purpose

FEVP v1.0 makes locality an auditable property of a floorplan editor. A report
must describe what was allowed to change, which pixels and labels remained
unchanged, whether the resulting topology is legal, and whether structural graph
consistency was preserved.

The protocol prevents three common mistakes:

1. Reporting an edit-quality metric without an outside-exactness audit.
2. Treating an unreported external metric as a failed or inferior result.
3. Replacing missing factors with imputed values.

## Mandatory Metrics

### Outside exact

`outside_exact` contains two required submetrics:

- `outside_pixel_exact`: strict `uint8` equality on the valid outside domain.
- `outside_room_label_exact`: strict semantic room-label equality on the same
  domain.

The outside domain excludes the declared edit region, the footprint exterior,
and the one-pixel dilation of the union of pre-edit and post-edit walls. No
tolerance is allowed.

### Validity

`validity` is the fpeval verification outcome for wall geometry, room faces, and
opening attachment.

### Conditional watertightness

`conditional_watertightness` is the mean watertight prediction rate over cases
whose post-edit target is watertight. Non-eligible cases use `null` in case rows
and do not enter the denominator.

### Node F1

`node_f1` is HouseMind-style Node F1 over unique room-type nodes.

### Edge overlap

`edge_overlap` is Jaccard overlap over unique room-type-pair adjacency edges.

## Optional Metrics

- `delta_iou`: predicted versus expected edit-change IoU.
- `ged`: HouseMind-style graph edit distance.
- `delta_mse`: predicted versus expected binary change-map MSE.

Optional metrics provide diagnosis and cannot replace any mandatory metric.

## Coverage States

Each metric uses one of:

- `reported_under_fevp`
- `reported_outside_fevp`
- `not_reported`
- `not_comparable`

A dash in a paper table maps to `not_reported` or `not_comparable`; it never means
that the method failed. Cross-dataset values cannot use `reported_under_fevp`
unless they share the complete FEVP contract.

## Report and Case Schemas

- `schema/fevp_report_v1.schema.json`
- `schema/fevp_case_row_v1.schema.json`

Reports require method, dataset, task, comparability, edit-region rules, scorer
identity, code commit, mandatory coverage states, source paths, and case counts.
Case rows require per-case exactness, validity, graph metrics, and explicit
watertight eligibility.

## Validation

```bash
uv run python scripts/validate_fevp_report.py \
  examples/fevp/complete_report_v1.json --require-complete
```

The validator checks:

- schema and protocol versions;
- required metadata and scorer provenance;
- metric status and value consistency;
- numeric ranges;
- source-map coverage;
- explicit N/A behavior;
- case-level conditional-watertight denominators.

## Reference Implementation

- Protocol validation and aggregation:
  `src/floorflow/evaluation/fevp_protocol.py`
- Edit metrics: `src/floorflow/evaluation/edit_detection.py`
- Topology: `src/floorflow/evaluation/fpeval_bridge.py`
- Graph metrics: `src/floorflow/evaluation/housemind_structure.py`
- Typed local routing: `src/floorflow/inference/typed_edit_graph.py`

`typed_edit_graph.py` is the reference typed-routing component. It is not the
complete FEVP scorer and does not define external-method behavior.

## Claim Boundary

Allowed:

> FloorFlow reports a complete FEVP v1.0 row in the evaluated matrix.

Not allowed without further evidence:

- FloorFlow is the first or only strict-locality system globally.
- An unreported external method is worse.
- HouseMind fails strict locality.
- Cross-dataset anchors form a leaderboard.

## Versioning

Any change to the outside domain, watertight denominator, graph matching,
selection rule, or scorer semantics requires a new FEVP protocol version.
Changing only an implementation must still update scorer commit and report
provenance.
