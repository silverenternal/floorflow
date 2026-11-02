# FloorFlow RPLAN 4096/512 Held-Out Screen

## Protocol

Cases are validation offset 256 through 767. RPLAN test is closed. FloorFlow
uses the five-seed-selected mask-only objective, mask-partition readout at
threshold `0.5`, batch 16, and eight epochs for five seeds. CE2EPlan uses its
released `b_model300000.pt` checkpoint on the same 512 cases.

This is not a matched-training comparison. CE2EPlan is a 300k-step, 1000-NFE
pretrained external anchor; FloorFlow is a 32,768-sample-exposure, one-forward
NFE candidate.

## Quality-First Comparison

| Method | Room IoU | Instance F1 | Type accuracy | Count MAE |
| --- | ---: | ---: | ---: | ---: |
| CE2EPlan released anchor | 0.6597 | 0.4814 | 0.7470 | **0.8652** |
| FloorFlow seed 37 | **0.6747** | **0.5088** | **0.7578** | 0.9688 |
| FloorFlow seed 43 | **0.6597** | **0.4885** | **0.7629** | 1.2090 |
| FloorFlow mean | **0.6672** | **0.4987** | **0.7603** | 1.0889 |

The original two-seed screen exceeded the anchor by Room IoU `+0.0076`,
instance F1 `+0.0173`, and type accuracy `+0.0134`, while count MAE was worse.
The required five-seed expansion is stricter:

| Metric | Five-seed mean | 95% t-CI | Seed range | Anchor | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Room IoU | **0.66327** | [0.65350, 0.67305] | 0.65429-0.67433 | 0.65967 | +0.00361 |
| Instance F1 | **0.50158** | [0.48974, 0.51342] | 0.49283-0.51746 | 0.48139 | +0.02019 |
| Type accuracy | **0.75388** | [0.74621, 0.76156] | 0.74435-0.76087 | 0.74697 | +0.00692 |
| Count MAE | 0.88672 | [0.81463, 0.95880] | 0.81641-0.94336 | **0.86523** | +0.02148 |

Seeds are `19, 29, 37, 43, 2905`. The seed-level interval for Room IoU extends
below the external anchor, so the IoU result is a positive mean trend against a
much larger-budget pretrained checkpoint, not a stable matched-budget win.
The one-query/one-primary-component readout reduces count MAE from `1.05195`
to `0.88672`, leaving only a small remaining gap to the anchor.

## Budget

| Property | FloorFlow | CE2EPlan anchor |
| --- | ---: | ---: |
| Parameters | 13.61M | 9.64M |
| Training sample exposures | 32,768 | 153.6M readme estimate |
| Declared optimizer steps | 256 | 300,000 |
| Sampling NFE | 1 | 1,000 |
| Room geometry | variable polygon | axis-aligned box |

The comparison establishes that the architecture is competitive with a much
larger-budget external anchor on the same validation cases. It does not establish
matched-budget superiority.

### Larger Held-Out Confirmation

The same two checkpoints were evaluated on validation offset 768 with 4,096
possible cases:

| Seed | Successful canonical cases | Room IoU | Instance F1 | Type accuracy | Count MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 37 | 4096/4096 | 0.6668 | 0.4991 | 0.7435 | 0.9187 |
| 43 | 4096/4096 | 0.6680 | 0.5064 | 0.7710 | 1.1335 |
| Mean | 4096/4096 | **0.6674** | **0.5028** | **0.7573** | 1.0261 |

Token round-trip exactness is `0.9934` for seed 37 and `0.9924` for seed 43.
No canonical decoding failures occurred. This larger disjoint slice confirms
that the 512-case result is not a small-sample artifact.

### Full Validation

The seed-37 checkpoint was evaluated over all 12,018 validation cases in four
disjoint shards:

| Metric | Value |
| --- | ---: |
| Room IoU | **0.66441** |
| Instance F1 | **0.51157** |
| Type accuracy | **0.74651** |
| Count MAE | **0.79198** |
| Canonical success | **12,018/12,018** |
| Token round-trip exact | **1.00000** |

This is the frozen full-validation quality reference for the 4096 mask-only
candidate. It is not a test result and is not directly comparable to a
different validation subset used for the CE2EPlan anchor.

## Safety-First Structure

The declared `partition_fill` readout was evaluated on the same 512 cases for
all five seeds:

| Metric | Mean | 95% t-CI | Seed range |
| --- | ---: | ---: | ---: |
| Validity | **1.0000** | [1.0000, 1.0000] | 1.0000-1.0000 |
| Conditional watertightness | **0.99023** | [0.98640, 0.99407] | 0.98633-0.99414 |
| Node F1 | **0.90225** | [0.89597, 0.90853] | 0.89450-0.90758 |
| Edge overlap | **0.60043** | [0.58463, 0.61623] | 0.59021-0.62088 |
| GED | **3.30508** | [3.20323, 3.40692] | 3.17773-3.40039 |
| Wall F1 | **0.58299** | [0.57977, 0.58621] | 0.58020-0.58587 |
| Dangling-endpoint rate | **0.01757** | [0.01620, 0.01893] | 0.01595-0.01861 |
| Room IoU | 0.64114 | [0.62775, 0.65453] | 0.62874-0.65278 |

All `2,560/2,560` seed-case evaluations assemble canonically. Opening F1 is
`1.0` because the supplied entrance condition is consumed by canonical
readout; it is explicitly not evidence for the learned opening head.

The safety-first mode trades quality-first Room IoU/F1 for topology.
Quality-first and safety-first numbers must not be mixed in one row.

## Bound

- No HouseMind comparison is claimed.
- No RPLAN test or leaderboard claim is made.
- CE2EPlan remains a pretrained external anchor, not a matched baseline.
- The 512-case quality and structure means cover five FloorFlow seeds.
- Learned wall/opening heads are evaluated separately from the conditioned
  canonical fallback.
