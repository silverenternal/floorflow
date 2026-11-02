# Third-party research references

These source trees are pinned for architecture study and adaptation. The main
project only imports a pinned class where the upstream license permits it.

The Git repository commits this provenance and license index only. Third-party
source trees, datasets, checkpoints, nested Git repositories, and other large
assets are intentionally excluded from version control. Clone the pinned
revisions listed below into `third_party/` when a workflow requires them.

| Project | Source | License | Use |
|---|---|---|---|
| LayoutBridge | `IndoorMapping-Lab/LayoutBridge` @ `e02974a` | MIT | Adapted DDI-style fusion, anisotropic tensor refinement, and frequency losses in `src/floorflow/layoutbridge_adapters.py`. |
| Mask2Former | `facebookresearch/Mask2Former` @ `9b0651c` | MIT | Adapted query-mask decoder, masked cross-attention and Hungarian mask assignment in `src/floorflow/room_mask_decoder.py`. |
| HouseDiffusion | `aminshabani/house_diffusion` @ `083c9c5` | GPL/ research-only | Conceptual reference for room-loop and incidence supervision; no source copied. |
| House-GAN++ | `NitsanMaman/HouseGanPlusPlus` @ `47b1c0f` | GPL/ research-only | Conceptual reference for graph-conditioned room message passing; no source copied. |
| VectorFloorSeg | `DrZiji/VecFloorSeg` | No license found | Reference only; no source copied. |
| TLC-Plan | `rosolose/TLC-PLAN` | No code/license supplied at clone time | Paper workflow reference only. |
| MitUNet | `aliasstudio/mitunet` @ `ade0aa6` | MIT; bundled CubiCasa weights are CC-BY-NC | Thin-wall segmentation, Tversky and Mix-Transformer/U-Net design reference. Pretrained weights are not used. |
| fpvec-lab | `Cyprinus12138/fpvec-lab` @ `b4b4a16` | MIT; CubiCasa-derived data non-commercial | Readout/reconciliation findings and watertight room-tiling evaluation reference. |
| MiTA | `QishuaiWen/MiTA` @ `95460f7` | MIT | Adapted route-only top-k attention in `floorflow.model.official_mita_route`; used by semantic element and recursive graph heads. |
| EoMT | `tue-mps/eomt` @ `aa017c3` | MIT | Dynamic query/mask and Hungarian assignment design reference for room and object queries. |
| MaskPLAN | `HangZhangZ/MaskPLAN` @ `4af728a` | MIT | GDMAE partial-layout masking reference; training-only coarse-layout perturbations. |
| PMT | `tue-mps/pmt` @ `0e80372` | MIT | Prompt-mask query adaptation reference for object and room layers. |
| Trinity | `Kohaku-Lab/Trinity` @ `51334c5` | MIT | Room-token denoiser and graph transport reference; adapted in `floorflow.model.trinity_contour_room`. |
| DHSA | `xiongsiheng/DHSA` @ `069d553` | Apache-2.0 | Exact-total dynamic token-budget algorithm adapted in `floorflow.model.dynamic_budget`. |
| FloorplanTransformation | `art-programmer/FloorplanTransformation` @ `929e7a3` | MIT | ICCV 2017 raster-to-vector. Borrow the PyTorch `IP.py` / `IP_gurobi.py` integer-programming wall/junction reconciliation as a deterministic post-readout pass; reuse the corner/icon/room readout taxonomy. |
| Raster2Seq | `Cornell-VAILab/Raster2Seq` @ `a6c4e27` | MIT | SIGGRAPH 2026 labeled-polygon-sequence reconstruction. Borrow deformable-query polygon decoder and `diff_ras` differentiable rasterizer for a polygon-sequence readout head; use as the vector baseline. |
| GSDiff | `SizheHu/GSDiff` @ `a02af02` | GPL-3.0 | Geometry-enhanced structural graph generation (AAAI 2025). Conceptual reference only; no GPL source is copied. |
| FloorplanGAN | `luozn15/FloorplanGAN` @ `92b8fa2` | MIT | Vector generator + wireframe discriminator + differentiable `renderer_g2v`. Borrow the generator-to-vector rendering loss for adversarial refinement; do not copy the discriminator verbatim. |
| CAGE | `ee-Liu/CAGE` @ `11cf0d9` | MIT + Commons Clause (non-commercial) | NeurIPS 2025 continuity-aware edge network. Design reference only for a native edge-centric wall representation and watertight-topology readout; no source is vendored into the project. |
| FloorSet | `IntelLabs/FloorSet` @ cloned 2026-10-25 | Apache-2.0 | Official chip-floorplanning dataset repo. Ships the 100-case `LiteTensorDataTest` (255MB) used for physics parity and the 80/20 smoke study; the 1M training tar is not included (HuggingFace unreachable here). |
| GraphGPS | `rampasek/GraphGPS` @ `2801570` | MIT | Local MPNN plus global attention separation, staged encoder/fusion modules, and graph-transformer recipe. Conceptual reference for `typed_channel_fusion.py`; no source copied. |
| GraphMAE | `THUDM/GraphMAE` @ `b14f080` | MIT | Masked graph feature/edge reconstruction and SCE loss. Design reference for future channel self-supervised warm starts; no source copied. |
| AdapterFusion | `adapter-hub/adapters` @ `53a1ea1` | Apache-2.0 | Independently trained adapters followed by a separately trained query-key fusion block. Reference for composable channel training scopes; no source copied. |
| PyHGT | `acbull/pyHGT` @ `85eaccd` | MIT | Typed node and relation projections with relation-specific attention. Reference for the existing relation-specific typed fusion; no source copied. |
| Perceiver IO | `lucidrains/perceiver-pytorch` @ `202cd4e` | MIT | Input-to-latent cross-attention and latent self-attention separation. Reference for cacheable channel encoding and a separable fusion stage; no source copied. |

When adapting MIT code, retain this notice and the upstream license.
The MIT-licensed `FloorplanTransformation`, `Raster2Seq` and `FloorplanGAN`
trees may be adapted by importing or vendoring specific classes only; the
GPL-3.0 `GSDiff` and Commons-Clause `CAGE` trees are read for design only and
no source is copied.
