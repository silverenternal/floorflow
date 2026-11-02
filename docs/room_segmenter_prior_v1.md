# Contour-only room-segmenter prior

## Purpose

The pretrained-free semantic flow head was under-trained for dense room
partitions.  This branch trains a dedicated contour-to-room anchor and exposes
its logits to the existing target-free overlay decoder.  The CFM model still
generates wall, opening and furniture layers; the anchor only supplies a dense
room-type distribution.

Inference uses only:

- the exterior footprint;
- a fixed style id from the train-only style bank;
- model-generated room, wall, opening and furniture layers.

No room control, coarse layout or target raster is available to the anchor.

## Architecture

- Torchvision ResNet18 encoder with ImageNet initialization during training.
- Three-channel deterministic contour representation: footprint, deeply eroded
  interior and boundary band.
- Full-resolution FPN-style decoder with 128 channels.
- Dense room-class output for the 24 CubiCasa room groups.
- 12,385,240 parameters.

The implementation is local:
`src/floorflow/heads/room_segmenter.py`.
The loader and taxonomy check are in
`src/floorflow/inference/room_segmenter_prior.py`.

## Data

The selected run consumes all 5,126 train floors and selects checkpoints on the
491 validation floors.  It does not consume test labels.  Rare-case resampling
was tested but lost to uniform visitation:

| Training branch | Data per epoch | Best val Room IoU |
| --- | ---: | ---: |
| ResNet18-FPN, 3 epochs | 1,024 | 0.08487 |
| ResNet18-FPN, 20 epochs, class-presence sampler | 5,126 | 0.14787 |
| ResNet18-FPN, 20 epochs, uniform | 5,126 | 0.15438 |
| Continued 20 epochs, uniform | 5,126 | 0.16224 |

The cache-weight lookup is reused during launch, and DataLoader workers are
persistent with preloading available.  These changes removed the repeated
decompression of all training shards before GPU execution.

## Evaluation contract

The anchor is evaluated through the unchanged target-free overlay path.
`--room-prior-weight 2 --retrieval-prior-weight 0` was selected on validation
data.  The contour-retrieval prior is disabled when it conflicts with the
trained anchor.

The 64-case validation screen produced Room macro IoU `0.13850`, instance
geometric IoU `0.48194`, room polygon F1 `0.16555` and coverage `0.91811`.
The full 491-case validation result is recorded under
`outputs/reports/roomseg_uniform40_w20_retr0_val491_v1.json`.

The remaining error concentration is captured by
`outputs/reports/roomseg_uniform40_w20_retr0_stratified_val491_v1.json`.

## License

The residual encoder is instantiated from Torchvision's BSD-licensed ResNet18
implementation.  No upstream weights are redistributed; the selected local
checkpoint is trained from the ImageNet initialization on CubiCasa train only.
