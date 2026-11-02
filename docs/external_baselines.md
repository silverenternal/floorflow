# External baseline assets and preflight

Status: assets and native-format smoke tests are complete. Comparable CubiCasa
results are **not** complete, because each external method expects a room
program or room graph rather than only a footprint.

## HouseDiffusion

- Official temporary checkpoint source:
  `https://drive.google.com/file/d/16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR/view`
- Local checkpoint:
  `outputs/external_baselines/house_diffusion/house_diffusion_model.pt`
- Checkpoint SHA-256:
  `7b22c916ec8d5fce087ce9bfe277817a93d60ca2eed9f8bbcf4b8eefec43797a`
- Released architecture: 26,541,330 parameters, 186 state tensors.
- Native RPLAN smoke set: six official RPLAN sample JSON files linked under
  `third_party/house-diffusion/datasets/rplan`.
- MagicPlan alternative recommended by the HouseDiffusion authors:
  155,229 layouts, 465,620 room masks, 2.5 GB extracted. The source archive
  SHA-256 is
  `800ead4c55596207a9e67c8d508b42fd85ab5385a2d63010edb6a958d21c4901`.
- The original full RPLAN corpus remains behind the authors' request form and
  is not redistributed here.

Preflight passed a strict state-dict load, RPLAN processing, and a finite model
forward pass with shapes `dec=(1,2,100)` and `binary=(1,16,100)`.

## House-GAN++

- Official checkpoint source:
  `https://raw.githubusercontent.com/ennauata/houseganpp/main/checkpoints/pretrained.pth`
- Local checkpoint: `third_party/houseganpp/checkpoints/pretrained.pth`
- Checkpoint SHA-256:
  `b87d783cbdf0cc20b5b9f96a235fb00eefd39b07d35f6ee334fff8f9157e8bba`
- Released architecture: 635,569 parameters, 50 state tensors.
- The official six-sample list was rewritten to local relative paths.
- Official six-sample inference completed with 12 artifacts in
  `outputs/external_baselines/houseganpp/smoke_r1`.

## MaskPLAN

- Official checkpoint source:
  `https://drive.google.com/file/d/1KkHRhd19TE-oIrgpPlY53TNGP1YT848j/view`
- Configuration: `All_Large_Single_cross`.
- Source archive SHA-256:
  `c941707b5f0eeaf18249c7947f1bbfd398593fe269d349c177f550b9c54b4f6a`
- Model: 388,897,374 parameters; TensorFlow checkpoint is 4.67 GB.
- Local files:
  `third_party/MaskPLAN/MaskPLAN_Trained/All_Large_Single_cross/`.
- Processed RPLAN arrays, VQ decoder/codebook, and 80,788 boundary images are
  present locally.
- A CPU TensorFlow 2.15 environment is isolated at
  `.external_envs/maskplan`.
- Recreate that environment with:

  ```bash
  uv venv .external_envs/maskplan --python 3.11
  uv pip install --python .external_envs/maskplan/bin/python \
    "tensorflow-cpu==2.15.1" "numpy<2" opencv-python-headless \
    shapely h5py tqdm matplotlib scipy pandas scikit-image networkx imageio
  ```

- Boundary-only (`Our I`) inference completed for site `16005`; raw and
  post-processed outputs are in
  `outputs/external_baselines/maskplan/smoke_r1`.
- A frozen 64-case native RPLAN boundary-only subset completed. Mean
  footprint coverage is `1.0`; mean generated pixels outside the rendered
  boundary are `614.72`. This is a native benchmark, not a CubiCasa result.

## Comparability boundary

HouseDiffusion and House-GAN++ consume room graphs and room masks. MaskPLAN
consumes a fixed-length RPLAN room-token program plus boundary tokens. None is
a drop-in contour-only CubiCasa baseline. Any comparison must state whether room
types/counts/adjacency came from an oracle program or from a separately trained
predictor. Native RPLAN and MagicPlan smoke results must not be reported as
CubiCasa test results.

## CubiCasa oracle-program adapters

`scripts/adapt_cubicasa_to_housegan.py` and
`scripts/adapt_cubicasa_to_housediffusion.py` convert CubiCasa room connected
components into the shared RPLAN JSON contract. The resulting programs are
explicitly marked `target_derived_control: true` and must only appear in the
oracle-program table.

- CubiCasa val64: 64 HouseGAN/HouseDiff programs exported.
- HouseDiffusion fits 58/64 cases under its 100-corner cap and passes a finite
  forward with `dec=(1,2,100)` and `binary=(1,16,100)`.
- House-GAN++ completes all 64 cases. Its measured oracle-program summary is
  `room_support_iou=0.13366`, `footprint_coverage=0.18626`,
  `mapped_class_iou=0.01429`, and mean `3254.38` generated pixels outside the
  target footprint.

The machine-readable result is
`outputs/reports/external_baseline_comparison_v1.json`. These low baseline
numbers are retained as evidence that graph-only external models do not solve
the contour-constrained task without adaptation.

The machine-readable inventory and preflight record is
`outputs/reports/external_baseline_availability_v1.json`.
