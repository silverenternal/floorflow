# FloorFlow

**Structured floorplan editing with explicit geometry and object continuity**

FloorFlow studies how to make a local edit to an existing floorplan while
preserving unaffected geometry and the identity of objects that survive the
edit. The repository includes learned floorplan generation and control
pipelines, an explicit geometry backend for typed edits, and evaluation
protocols for measuring what changed and what was preserved.

## From an edit request to a committed floorplan

1. **Represent.** Store rooms, wall material, openings, fixtures, and the
   footprint as separate geometry layers. The rendered plan is a view of this
   state.
2. **Bind and apply.** Resolve a typed request against the committed input
   state. A room edit can use explicit request geometry or a saved model
   proposal. The guarded proposal path checks the source-state fingerprint,
   and the permission adapter constrains predicted split geometry to the
   allowed region. Opening addition and replacement require an explicit wall
   host that resolves in the input state.
3. **Track.** A room reclassification retains the room ID. A split records
   the retained room and its new child. A merge records the absorbed ID as an
   alias of the survivor. These runtime IDs describe continuity within an
   edit history. A fresh decode derives IDs from the input namespace and
   geometry, with no cross-decode identity guarantee.
4. **Verify.** An optional 2D R2010 DXF package includes a JSON sidecar, a
   transition archive, and a receipt. The repository reader checks file and
   history linkage against caller-supplied hashes and compares reconstructed
   geometry and metadata with the source state.

The [structured editing API](docs/cad08_structured_editing.md) documents the
transaction inputs, guarded variants, and export contract.

## Run the CPU example

From the repository root, on a shell with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cad]'
demo_dir=$(mktemp -d)
python examples/floorplan/cad08_quickstart.py "$demo_dir"
```

The [example](examples/floorplan/cad08_quickstart.py) creates a synthetic
floorplan, splits one room, and reclassifies the new room by its runtime ID.
It checks that the reclassification preserves the child geometry and the
remaining room, then exports and reads back a two-transition package. On
success it prints the checks and the path to `plan.dxf`, `metadata.json`,
`history.json`, and `receipt.json`. This path runs without a dataset, GPU, or
model checkpoint.

To run the focused CPU suite in the same environment:

```bash
python -m pip install 'pytest>=8'
python -m pytest -q tests/test_cad08_*.py
```

The example and tests cover synthetic geometry and readback with the repository's
own reader. Interoperability with third-party CAD tools is outside that test scope.

## Evaluation protocols and evidence

The results below have different inputs and units of analysis. Each report
records the conditions under which its results apply.

| Protocol | Inputs and evaluation scope | Tracked evidence |
| --- | --- | --- |
| Oracle-control editing | Pre-edit plan, a typed operation, a declared edit region, and target-derived pre/post controls. 2,479 paired edit requests across five operations on the development-exposed CubiCasa5K test503 split. Repeated seeds use the same requests. | [Protocol and per-operation metrics](docs/fp_oracle_control_protocol_status_v1.md) · [Floorplan-level intervals and paired comparisons](outputs/reports/oracle_submission_statistics_v2.json) |
| Predicted-control validation | Controls predicted without access to target annotations at inference. Fidelity to annotated layouts and structural plausibility are evaluated on CubiCasa5K val491. The plausibility report covers five repeated seeds. | [Fidelity](outputs/reports/predicted_control_fidelity_val491_v3.json) · [Plausibility](outputs/reports/predicted_control_plausibility_fiveseed_v12.json) |
| Predicted-operation audit | The robustness audit covers 738 development-exposed test prompts, including 438 executed and 300 abstained. Semantic success is reported separately on supported prompts from an older val191 model run. | [Prompt outcomes, denominators, and failure reasons](outputs/reports/predicted_operation_evidence_v2.json) |

The [strict locality protocol](docs/fp_strict_locality_protocol_v2.md) defines
what counts as unchanged outside the declared edit region. Edit effect,
preservation, and geometric validity are reported as separate quantities. The
[freeze manifest](configs/floorplan_submission_evidence_freeze_v2.json)
records report provenance and scope. The tracked reports support inspection of
the reported results. Learned-pipeline reruns require external data and
run-specific checkpoints.

## Code and data

| Area | Entry points |
| --- | --- |
| Geometry and typed transactions | [State representation](src/floorflow/representation/cad08_state.py) · [Transaction API](src/floorflow/inference/cad08_transaction.py) · [History export](src/floorflow/representation/cad08_history_export.py) |
| Learned pipeline | [Architecture](docs/architecture.md) · [Preprocessing](scripts/preprocess.py) · [Training](scripts/train.py) · [Generation](scripts/infer.py) · [Layered evaluation](scripts/evaluate_layered_pipeline.py) |
| Data and dependencies | The learned pipeline uses CubiCasa5K SVG data, placed locally under `.data/cubicasa5k`. See the [third-party components inventory](third_party/README.md). |

To build a local cache from the SVG dataset, run the preprocessor from the
repository root after placing the dataset at the path above:

```bash
python scripts/preprocess.py --data-root .data/cubicasa5k \
  --cache-root outputs/cache/cubicasa5k_full_v2 --splits train val test \
  --image-size 256 --room-min-count 100 --furniture-min-count 100
```

Training and learned evaluation use the optional `train` and `eval`
dependencies. The layered evaluator derives coarse controls from annotations
when no serialized control bundle is provided.
