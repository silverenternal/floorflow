# Raster2Seq Identity Adapter Diagnostic Definition v1

This diagnostic checks whether a project-defined official-style Raster2Seq input renderer changes reconstruction metrics on 503 unique CubiCasa test503 inputs. It is not an editing experiment and it is not an official matched reconstruction result.

## Primary metrics

- Room macro IoU and matched-instance F1.
- HouseMind-style Node F1, Edge overlap, and GED.
- Wall, opening, and furniture binary F1.
- Pixel accuracy over the footprint.

## Identity diagnostics

- Identity Delta-IoU and Delta-MSE test no-change output stability.
- Both are trivial when the same deterministic output is compared with itself and cannot replace reconstruction quality.
- Raster2Seq replay exactness against available paired predictions is reported as a provenance/stability diagnostic.

## Boundary

- Raster2Seq remains reconstruction-only.
- The adapter result cannot be called native, official, matched, or reconstruction SOTA.
- FloorFlow identity reconstruction uses oracle controls.
- No FEVP editing claim follows from this audit.
