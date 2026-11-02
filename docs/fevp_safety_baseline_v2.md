# Validity and Watertightness Safety Boundary

The five-check validity is 0.9797, and watertightness is used as an auditable safety baseline. The guard changes conditional watertightness from 0.9636 to 0.9626, repairing 57 failures while introducing 67 regressions; it is not claimed as a net method improvement.

## Allowed

- Validity is reported as a five-check conjunction.
- Watertight failures are localized with hole and dangling-edge diagnostics.
- The guard's repair and regression counts are both disclosed.

## Forbidden

- Validity is 1.0 under the hardened five-check definition.
- The watertight guard provides a net improvement.
- Watertightness is a primary contribution of the method.
- The guard should be promoted based on repaired cases alone.
