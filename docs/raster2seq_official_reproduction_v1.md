# Raster2Seq Official Reproduction v1

Date: 2026-11-01

## Checkpoint

- Repository: `haopt/Raster2Seq`
- Model: `cubicasa5k`
- Checkpoint: `third_party/Raster2Seq/checkpoints/cubicasa5k/checkpoint.pth`
- Size: 1.4 GB
- SHA-256: `f7c0ef9379aa8cf11349da8b2179214e0d495af758383984a2361a1309aab3b7`
- Config SHA-256: `6367f59326b4c8acdb046b0f08be15b1d62dd20d8fe364ad0061d7a3783ccefc`
- Published CubiCasa5K RoomF1: 88.7

## Runtime Contract

The repository contains CPython 3.11 builds of:

- `models/ops/MultiScaleDeformableAttention*.so`
- `diff_ras/native_rasterizer*.so`

Both directories must be on `PYTHONPATH`; without the deformable-attention
build, model construction fails before checkpoint loading.

## Official Smoke

Command:

```bash
PYTHONPATH=third_party/Raster2Seq/models/ops:third_party/Raster2Seq/diff_ras \
  .venv/bin/python third_party/Raster2Seq/predict.py \
  --debug --dataset_name cubicasa \
  --dataset_root outputs/cache/raster2seq_cc5k_val_v2/val \
  --checkpoint third_party/Raster2Seq/checkpoints/cubicasa5k/checkpoint.pth \
  --output_dir outputs/reports/raster2seq_official_smoke \
  --semantic_classes 12 --input_channels 3 --poly2seq \
  --seq_len 512 --num_bins 32 --disable_poly_refine \
  --dec_attn_concat_src --per_token_sem_loss --use_anchor \
  --ema4eval --save_pred --device cuda:0 --num_workers 0
```

Observed:

- model parameters: `16,771,089`;
- one CubiCasa validation case completed on `cuda:0`;
- inference time: about `12.0 s`;
- output: `9` semantic polygons total, including `5` room polygons and
  opening polygons;
- saved JSON, NPY, predicted room map, and predicted floorplan visualization.

The checkpoint loads with only expected decoder KV-cache buffers reported as
unexpected for inference mode.

## Claim Boundary

This proves that the official public checkpoint and inference model are
executable in the local environment. It is not yet a matched room metric row,
teacher model, or SOTA comparison. The next audit must fix the common raster
coordinate transform and evaluator before using its polygons as evidence.

Evidence:

- `outputs/reports/raster2seq_official_smoke/cubicasa5k/jsons/03510.json`
- `outputs/reports/raster2seq_official_smoke/cubicasa5k/03510_pred_floorplan.png`

