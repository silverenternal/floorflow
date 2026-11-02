# Adversarial Edit-Prompt Proxy Robustness

| Arm | Outside exact | Violation | Validity | Delta-IoU | Node F1 | Edge F1 | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| typed_edit_graph | 1.000000 | 0.000000000 | 1.000000 | 0.930502 | 0.944675 | 0.862417 | 2.622500 |
| dense_cross_attention | 0.999484 | 0.000516062 | 1.000000 | 0.860200 | 0.942304 | 0.858898 | 2.690000 |
| no_edit_gating | 0.984665 | 0.015335273 | 1.000000 | 0.824777 | 0.937453 | 0.845819 | 2.932500 |
| no_structural_graph | 1.000000 | 0.000000000 | 1.000000 | 0.861138 | 0.942304 | 0.858898 | 2.690000 |

| Prompt proxy | Cases | Delta-IoU typed | GED typed | Violation typed | GED no-gate |
|---|---:|---:|---:|---:|---:|
| ambiguous_global_scope: Make the whole layout more open and coherent. | 100 | 0.942085 | 5.570000 | 0.000000000 | 6.000000 |
| enclosed_room: Modify the enclosed room while keeping every adjacent room fixed. | 100 | 0.934396 | 3.460000 | 0.000000000 | 3.670000 |
| occupied_room_insertion: Insert a room where existing room support already occupies the area. | 100 | 0.878873 | 1.360000 | 0.000000000 | 1.960000 |
| opening_support_conflict: Delete the opening/support segment while preserving room connectivity. | 100 | 0.966652 | 0.100000 | 0.000000000 | 0.100000 |

The prompts are conflict-mode proxies attached to real controls. They do not evaluate a natural-language parser.

The paired tests compare typed against dense, no-gate, and graphless on the same rows.
