# Room Page-Floor Five-Seed Confirmation Protocol v1

Date: 2026-11-01

## Purpose

The room mechanism passed two-seed train1024 promotion. This protocol is the
next promotion gate: full-train confirmation before the room readout can be
frozen and connected to wall, junction, and adjacency consumers.

## Data Contract

- CubiCasa training split after empty-target filtering: `5,122` floors.
- Calibration: validation offset `0`, `384` cases.
- Disjoint holdout: validation offset `384`, `107` cases.
- Validation after empty-target filtering: `491` cases.
- Calibration and holdout do not overlap.
- The test split remains closed.

The four removed empty training targets remain explicitly reported:

```text
high_quality_architectural/1128__Floor-2
high_quality_architectural/675__Floor-1
high_quality_architectural/12661__Floor-1
high_quality/6555__Floor-1
```

## Mechanism

The candidate is the only mechanism that has passed two-seed promotion:

```text
page-floor crop
+ canonical parent warm-start
+ encoder adaptation
+ query-set head
+ connected query-count selection gate
```

All seeds use the same canonical warm-start checkpoint
`outputs/train/room_pagefloor_querygate_screen256_seed19_v1/best.pt`, preserving
the parent while varying the training seed. The baseline is the static parent
`outputs/train/room_program_supervision_only_fast64_v2/best.pt`.

## Readout Selection

Training selects the checkpoint epoch and query blend on calibration384 only.
The selected blend is then frozen before evaluation on disjoint holdout107.
No holdout or test value participates in checkpoint selection.

## Seeds

```text
19, 23, 29, 31, 37
```

All seeds use eight epochs, batch size eight, learning rate `2e-4`, encoder
learning rate `5e-5`, and the same loss/readout contract as the train1024
promotion.

## Statistics

- Each seed compares candidate and static parent on the same case.
- Primary per-seed intervals use a 5,000-iteration paired percentile
  bootstrap.
- Wilcoxon signed-rank p-values use one paired difference per case.
- Rank-biserial correlation is the unsigned-rank paired effect size.
- Cross-seed pooling clusters repeated seed rows by case before bootstrap.
- A pass requires at least `4/5` seeds with the beneficial direction and an
  interval excluding zero for Room IoU, instance F1, and count MAE.
- Type accuracy and room support must not regress in the seed mean.

The case-clustered pooled test is descriptive because it reuses the same cases
across seeds. It does not turn seeds into independent population samples.

## Reproduction

```bash
.venv/bin/python scripts/run_room_pagefloor_fiveseed_confirmation_v1.py \
  --seeds 19 23 29 31 37 \
  --device cuda:0 \
  --precision bf16 \
  --skip-existing

.venv/bin/python scripts/build_room_pagefloor_fiveseed_confirmation_v1.py
```

Expected outputs:

- `outputs/reports/room_pagefloor_confirmation_manifest_v1.json`
- `outputs/reports/room_pagefloor_confirmation_fiveseed_v1.json`
- `paper/tables/room_pagefloor_confirmation_fiveseed_v1.md`

## Current State

The full-train runner, disjoint offset selection, evaluator, and five-seed
aggregator are implemented and tested. CPU contracts completed training and
evaluation at validation offset `384`. The GPU run is pending because a separate
chips experiment currently occupies `cuda:0`; this run will not be launched
against concurrent chip work. The runner queries `nvidia-smi` and refuses to
start on an already occupied device unless `--allow-concurrent-gpu` is explicitly
provided.
