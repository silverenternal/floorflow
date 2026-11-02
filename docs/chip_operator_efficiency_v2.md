# Chip Operator Efficiency v2

Date: 2026-11-01

## Decision

The remaining Official-100 bottleneck is not a sparse-attention microkernel.
It is single-case refiner execution with one case per block-count group.

Exact block-range sharding changes scheduling only:

- model checkpoint unchanged;
- NFE unchanged;
- refiner steps and optimizer unchanged;
- legalizer settings unchanged;
- per-case RNG seed unchanged;
- output rows keyed by the original case index.

## Measured Result

| Arm | Serial wall seconds | Concurrent shard wall seconds | Speedup |
|---|---:|---:|---:|
| Candidate | 2023.27 | 1014.88 | 1.99x |
| Trinity d192 | 2395.55 | 1250.03 | 1.92x |

The candidate profile remains refiner-heavy:

- sampling: about 49 seconds on the measured 100-case shard;
- refiner: about 1954 seconds;
- decode: about 1.6 seconds;
- scoring: about 17 seconds.

Adding another sparse-attention kernel cannot remove that gap. The useful
runtime change is sharding independent block ranges while preserving exact
case-level decisions.

## Operating Constraint

The shard scheduler is an execution optimization, not a changed inference
budget. Reports must state:

- number of shards;
- GPU device;
- per-shard wall time;
- merged wall time;
- the exact merge utility.

## Artifacts

- `scripts/build_chip_operator_profile.py`
- `scripts/merge_chip_portfolio_shards.py`
- `scripts/merge_chip_official_shards.py`
- `outputs/reports/chip_operator_profile_v2.json`

## Boundary

The 2x wall-time reduction depends on running independent shards concurrently
on the same GPU. It does not reduce model FLOPs, NFE, refiner work or legalizer
work and must not be presented as an equal-compute efficiency result.
