# CubiCasa data and strata provenance

The official CubiCasa5K repository is pinned at:

- repository: `https://github.com/CubiCasa/CubiCasa5k`
- commit: `c34440266665a11f4484eb06cd2e4b7d72ad76c1`
- local path: `third_party/CubiCasa5k`
- license: CC BY-NC 4.0, research/non-commercial use

Current data audit:

| Split | Cases | Colorful | High quality | Architectural | Coarse strata | Fine strata | Rare cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5126 | 199 | 1011 | 3916 | 166 | 4968 | 544 |
| val | 491 | 62 | 124 | 305 | 83 | 487 | 51 |
| test | 503 | 85 | 73 | 345 | 86 | 498 | 60 |

Each index row contains room class count, room component count, rare-pixel
counts, opening/furniture pixels, footprint area/fill/aspect/boundary and
curriculum strata. The generated audit is
`outputs/reports/cubicas_strata_v2.json`; per-case rows are in
`outputs/data/cubicas_strata_v2.jsonl`.

The official `floortrans/loaders` parser was reviewed for room/icon/instance
semantics. FloorFlow keeps its deterministic local rasterizer because the
current benchmark contract depends on its exact label and footprint mapping.
