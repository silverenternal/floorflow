# FEVP Claim Language v1

## Allowed

> Under FEVP v1.0, FloorFlow is the only method in the evaluated coverage matrix with a complete report for all five mandatory metrics.

> HouseMind provides a complementary unified MLLM interface; its public paper/demo does not expose a matched FEVP strict-locality audit.

> Raster2Seq provides a matched reconstruction reference; missing editing and FEVP fields are shown as N/A.

> Raster2Seq's official RoomF1 and CE2EPlan's RPLAN FID provide external reproducibility anchors, but neither enters a matched editing leaderboard.

> On CubiCasa test503 with declared oracle controls, FloorFlow reaches
> Delta-IoU 0.956703-0.999801 and Delta-MSE 0.000019-0.000096. These values
> numerically exceed the published HouseMind-E anchors 0.608 and 0.0019 under
> aligned metric definitions, while the datasets and input contracts differ.

## Required N/A definition

> A dash means not reported under FEVP or not comparable; it does not mean zero, failure, or inferior performance.

## Forbidden

- HouseMind fails strict locality.
- HouseMind or CE2EPlan is worse because a FEVP cell is missing.
- The matrix is a cross-dataset leaderboard.
- FloorFlow is the first strict-locality system globally.
- CE2EPlan is a matched editing baseline.
- The Delta-IoU comparison is a HouseMind RPLAN leaderboard rank.
- FloorFlow's aligned Node/Edge/GED exceed HouseMind-E.
