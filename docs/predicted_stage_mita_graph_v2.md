# Predicted Stage MIta Graph v2

Date: 2026-11-01

Superseded by `docs/predicted_channel_source_router_v3.md`; retained as the
matched control.

## Audit

The predicted-channel model now has a balanced local stage head for rooms,
walls, openings, and furniture. The remaining gap is different from the room
head gap:

| Metric family | Current mechanism | Missing evidence |
|---|---|---|
| Room identity | query decoder plus owner-preserving readout | absolute Room IoU remains 0.31 |
| Walls | local residual plus room-conditioned pixel gate | no full-image sparse routing |
| Openings | local residual plus wall-conditioned pixel gate | no global host-wall arbitration |
| Furniture | shape head plus room conditioning | no image-level relation routing |

The first MIta integrations cover sparse control, the family-stream mixer, the
recursive region graph, and masked query heads. That coverage was incomplete:
the four CAD semantic stage heads still used local convolutions only.

## Stage Routing

MIta is now used in the wall, opening, and furniture stage heads, and as a
causal global branch over already-materialized CAD stages. The room stage is
intentionally excluded while `preserve_room_stage=true`, because that readout
is replaced by the separately confirmed room model.

Each routed stage performs:

```text
field + prior + predecessor condition
-> local residual
-> 8x8 route tokens
-> MiTA sparse global memory
-> sample/stage-conditioned router budgets
-> full-resolution gated residual
-> stage logits
```

The route memory includes the current hidden state, the stage prior, the
predecessor condition, and the CAD stage token. Router budgets are predicted
from the pooled stage state and stage token. Training uses a differentiable
soft budget strength while the attention itself receives hard budgets; this
keeps top-k execution explicit without blocking gradients to the budget head.

The CAD action graph keeps its exact predecessor-mask attention and adds a
causal MIta residual over all earlier materialized stages. Later stages are
never visible, so the ordered CAD action sequence does not become an
unconstrained all-stage attention block.

## Why Not Make Every Attention Sparse

The eight CAD action tokens retain their exact dense predecessor attention;
MIta is added only as a residual causal memory. The room stage remains dense
while it reuses the separately confirmed room readout. MIta is applied where
the missing receptive field is real: full-image weak semantic stage evidence
and cross-stage CAD history.

## Capacity

The new candidate uses 128-wide stage heads and MIta routing on three weak
stages plus the causal CAD history. With the current CubiCasa taxonomy it has
about 4.21M parameters:

| Group | Parameters |
|---|---:|
| Field encoder | 278,720 |
| Typed channel head | 881,158 |
| CAD action graph | 253,010 |
| Rooms stage | 400,730 |
| Walls stage | 797,585 |
| Openings stage | 797,211 |
| Furniture stage | 804,267 |

The added capacity lands on the metrics that remain weak. The shared field and
channel backbones are unchanged from v1.

## Training Budget

| Stage | Cases | Epochs | Batch | Exposures | Updates |
|---|---:|---:|---:|---:|---:|
| Screen | 512 | 16 | 4 | 8,192 | 2,048 |
| Promotion | 1,024 | 20 | 8 | 20,480 | 2,560 |
| Confirmation | 5,126 | 20 | 16 | 102,520 | 6,420 |

The screen remains small enough for iteration. It is longer than v1 because
the route residual and dynamic budget branch need enough updates to leave
their near-identity initialization.

## Required Controls

1. v2 graph without stage MIta.
2. Equal-parameter local convolution control.
3. Dense global convolution or attention control.
4. Static MIta budget control.
5. Four heads with no predecessor conditioning.

Every arm reports room, wall, opening, furniture, Node, Edge, GED, validity,
watertightness, route entropy, and effective expert count. A win that appears
only on calibration is rejected.

## Files

- `src/floorflow/heads/dynamic_stage_refiner.py`
- `src/floorflow/model/cad_stage_heads.py`
- `src/floorflow/model/cad_stage_readout.py`
- `src/floorflow/model/predicted_channel_model.py`
- `configs/predicted_channel_dynamic_graph_v2.json`

No GPU run is claimed in this document.
