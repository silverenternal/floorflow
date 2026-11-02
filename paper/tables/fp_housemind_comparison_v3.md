# HouseMind Comparison v3

HouseMind is a complementary unified MLLM framework for floorplan understanding, generation, and editing. It is not a matched FloorFlow baseline because the official inference pipeline is not released.

| Dimension | HouseMind | FloorFlow | Comparison boundary |
|---|---|---|---|
| Input | Natural-language instruction and multimodal context | Control-gated local delta with an explicit edit region | Different interfaces |
| Unified scope | Understanding, generation, and editing | Local editing protocol plus completion evidence | Complementary contributions |
| Public demo | Runnable online demo listed | Reproducible local experiment package | Different execution surfaces |
| Official code | Project page marks `Code (Soon)` | Source, tests, rows, and hashes packaged | HouseMind cannot be rerun |
| Localized editing | Qualitatively claimed | Explicitly constrained and measured | Not a same-protocol comparison |
| Outside exact | No matched quantitative audit found | Pixel and room-label exactness reported | FloorFlow-only metric |
| Validity / watertightness | Geometric validity discussed; no matched table for strict editing | Explicitly audited | Different metrics and data |

## Allowed wording

> HouseMind and FloorFlow target complementary aspects of architectural floorplan generation. FloorFlow contributes a strict-locality editing protocol and verifiable evidence suite.

## Forbidden wording

- FloorFlow beats HouseMind's complete method.
- HouseMind fails strict locality.
- The two models rank on one HouseMind leaderboard.
- Raster2Seq is a native editing-aware method.

## Sources

- Project: <https://housemind.github.io/>
- arXiv: <https://arxiv.org/abs/2603.11640>
- Demo: <https://housemind.ai-structure.com/>
- CVPR page: <https://cvpr.thecvf.com/virtual/2026/poster/37878>
