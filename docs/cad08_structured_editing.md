# CAD08 structured editing API

This optional CPU layer stores a floorplan as explicit geometry records instead
of treating the visible raster as the full state. Its implementation is
separate from the learned floorplan-generation pipeline.

## State and transactions

`LayeredCad08State.from_decoded(...)` initializes separate room, opening,
fixture, wall-material, and footprint geometry. Initial object IDs are
system-created runtime IDs; they are not author-supplied identity and do not
promise identity across a fresh decode. Split and merge operations record
lineage, merge aliases, and touched fixture-to-room relations. Unresolved
relations remain `unknown`.

`floorflow.inference.cad08_transaction.apply_transaction` supports the typed
room and object transaction API. Binding uses only the committed-before state.
Proposal-backed room edits are distinct from direct-action controls. The
state-guarded and permission adapters additionally require a matching source
state fingerprint and constrain predicted split geometry to the bound source
and explicit split permission.

Opening `add` and `replace` require an explicit host ID that resolves uniquely
in the committed-before state. The new aperture must be valid, inside the
footprint, covered by that host, and non-overlapping. Replacement keeps the
opening ID; failures return an unchanged state with a reason. The transaction
does not infer a host ID from an image or choose one by nearest geometry.

## R2010 history package

The opt-in history exporter binds an external transition archive and supplied
trusted hashes to a two-dimensional R2010 DXF, JSON sidecar, and receipt. The
reader checks receipt/file hashes, archive linkage, supported geometry, and
metadata. These checks are separate from permissions, semantic edit success,
cross-rebuild identity, and the original E/P/G/J evaluation fields. A package
exported successfully is marked `exported_unvalidated` until a trusted receipt
and readback are supplied.

The export and readback modules use the bundled `ezdxf` and Shapely libraries;
they are not independent geometry implementations. The test fixtures are
synthetic CPU cases and do not establish acceptance by another CAD product.

## Installation and focused verification

```bash
uv sync --extra dev --extra cad
uv run pytest \
  tests/test_cad08_cpu_import.py \
  tests/test_cad08_integrated_v1.py \
  tests/test_cad08_permission_v1.py \
  tests/test_cad08_opening_lifecycle.py \
  tests/test_cad08_geometry_normalized_preservation.py \
  tests/test_cad08_history_archive.py \
  tests/test_cad08_history_receipt.py \
  tests/test_cad08_transition_history.py \
  tests/test_cad08_history_export.py \
  tests/test_cad08_history_dxf_readback.py -q
```

The CAD extra supplies Shapely 2.1 or newer and ezdxf 1.4.5 or newer in the
1.x series. The dev extra supplies pytest. The CAD import path is lazy, so these
CPU transaction and serialization tests do not require the Torch training
runtime. No checkpoint or dataset is required.
