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

The exporter and reader both use the optional ezdxf and Shapely dependencies.
The CPU tests exercise synthetic geometry. Interoperability with another CAD
product is outside that test scope.

## Installation and focused verification

Follow the [CPU quickstart](../README.md#run-the-cpu-example) to run a room
split and reclassification through the transaction, history, and readback
APIs. The same section gives the command for the full CAD08 CPU suite.

The CAD extra supplies Shapely 2.1 or newer and ezdxf 1.4.5 or newer in the
1.x series. The CAD import path is lazy. The example and focused tests run
without the Torch training runtime, a dataset, or a checkpoint.
