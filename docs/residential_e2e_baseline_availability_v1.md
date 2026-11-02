# Residential E2E Baseline Availability v1

Status: `passed`

| Method | Track | Claim state | Reason |
|---|---|---|---|
| Raster2Seq | cubicasa_reconstruction | not_comparable | Official CubiCasa reconstruction anchor is reproduced, but no matched FloorFlow reconstruction run shares its 509-case contract. |
| CE2EPlan | rplan_generation | context_only | Official RPLAN generation anchor is reproducible locally, but no matched FloorFlow RPLAN model or shared scorer exists. |
| HouseMind | instruction_editing | unavailable | Weights are available, but official tokenizer/serialization/inference/post-processing contracts remain unreleased. |
| HouseMind | rplan_generation | unavailable | Weights are available, but official tokenizer/serialization/inference/post-processing contracts remain unreleased. |
| RenoLayout | renovation_editing_reference | context_only | No matched executable renovation inference package is available. |
| TLC-Plan | rplan_generation | unavailable | The public repository currently exposes README/assets without executable training or inference code. |
| FMLM | rplan_generation | unavailable | No executable code or checkpoint is advertised. |

Only `matched` rows with the same split, conditioning, preprocessing, output contract, and scorer can enter a leaderboard.

External SOTA rows enabled at this gate: `0`.
