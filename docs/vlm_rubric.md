# Floorplan VLM Review Rubric

The local evaluator is `Qwen3-VL-8B-Instruct`. It receives only the generated drawing and returns strict JSON.

## Scores

Each dimension is scored from 1 to 5:

- `room_plausibility`: rooms form plausible functional spaces and coherent regions.
- `topology_consistency`: doors/windows attach to walls and rooms form a navigable partition.
- `furniture_room_consistency`: furniture types are contained in compatible rooms.
- `drawing_readability`: walls, rooms, openings and symbols are distinguishable at normal viewing size.

## Current 100-sample result

| Dimension | Mean |
|---|---:|
| Room plausibility | 1.66 |
| Topology consistency | 1.66 |
| Furniture-room consistency | 1.35 |
| Drawing readability | 1.49 |

All 100 generated-only responses produced parseable JSON. The low scores confirm that the current model is not yet design-grade; these scores are lower than the earlier panel-based run, which was contaminated by reference images in the comparison panels.

Objective consistency against deterministic metrics is available at
`outputs/eval/vlm_rubric/consistency_generation_only.json`.

## Limitations

- VLM scoring is not a replacement for the deterministic structural gate.
- A second independent rater is still required for inter-rater agreement.
- Prompt/rubric version must be frozen before the final ablation matrix.
