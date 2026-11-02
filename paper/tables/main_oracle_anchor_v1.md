# Main Oracle-Control Editing Anchor v1

Status: frozen candidate. CubiCasa5K test503, declared oracle controls, eight
typed edits per floor, five operations, five repeated seeds.

| Operation | N | Delta-IoU | Delta-MSE | Outside exact | FP-Eval validity |
|---|---:|---:|---:|---:|---:|
| room type | 487 | 0.999199 | 0.000069 | 1.000000 | 1.000000 |
| room add | 495 | 0.999801 | 0.000030 | 1.000000 | 1.000000 |
| room remove | 500 | 0.999756 | 0.000019 | 1.000000 | 1.000000 |
| opening delete | 502 | 0.956703 | 0.000096 | 1.000000 | 1.000000 |
| furniture delete | 495 | 0.987812 | 0.000083 | 1.000000 | 1.000000 |

Independent FEVP five-check validity is `0.9124-0.9569`. Conditional
watertightness is `0.9579-0.9675`.

HouseMind-E publishes Delta-IoU `0.608` and Delta-MSE `0.0019`. FloorFlow
numerically exceeds both anchors, but the datasets and input contracts differ.
This is not a HouseMind leaderboard rank.
