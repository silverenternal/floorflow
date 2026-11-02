# Predicted-Control Zero-Path Audit v1

## Scope

Checkpoint:
`outputs/train/typed_channel_v6_zero_route_consumer_contract_v3/best.pt`

SHA-256:
`a994801773dc6bebd46b91738a0992c33d41eacede712bcd86fbaace8d1278b6`

The audit separates intentional residual dead starts from permanently
unreachable paths. A parameter is a defect only when the remaining pipeline
cannot make it live or when a consumer never reads its output.

## Checkpoint Scan

The checkpoint contains two multi-element all-zero matrices in active model
code:

| Parameter | Classification before repair |
| --- | --- |
| `field.evidence_fusion.2.weight` | Permanently zero under `room_adapter` because its producer was learned but the final projection was outside the training scope. |
| `action_graph.route_budget.program.2.weight` | Permanently zero because the program stage has one predecessor and the budget formula multiplied the learned logit by `top_k - 1`, which is zero. |

Zero biases in normalization layers and zero biases in direct residual heads are
not counted as dead paths: the weight path or downstream projection can make
them live.

## Repairs

### Field evidence

`room_adapter` now includes `field.evidence_fusion.2.*`. The first evidence
fusion layer in the retained checkpoint is learned and non-zero, while the final
projection was zero. Training the final projection activates the existing
evidence summary without reopening the frozen field encoder.

### CAD program route

`CadActionGraph` now uses the following single-predecessor budget:

```text
budget = 0.5 + 0.5 * sigmoid(logit)
```

instead of the constant `1 + sigmoid(logit) * (top_k - 1)`. The soft budget is
also passed into `OfficialMiTARouteAttention` and scales the selected route
representation, so the budget head has a differentiable consumer.

The gated program action token now enters the room token before room-class
fusion. This connects the program stage to room class/mask identity instead of
leaving the program route as an isolated diagnostic.

The differentiable budget and program-token room fusion are opt-in through
`differentiable_route_budget` and `program_room_action_fusion`. Both default to
false so historical checkpoint inference remains unchanged; both are enabled in
`configs/predicted_channel_v6_room_identity_adapter_joint.json`.

### Query-set cold starts

`RoomQuerySetHead.presence`, `count`, and `class_count` final projections now use
a near-zero non-zero initialization (`std=0.001`) rather than exact zero. This
keeps the initial outputs close to the prior while allowing set-attention,
global-summary, and MIta route heads to receive first-step gradients.

`RoomMaskClassFusion.program_gate` now starts at `0.001` rather than exact zero,
for the same reason.

The room mask output projection now uses `std=0.001` instead of exact zero.
Class supervision alone was able to shape query features while the spatial mask
path waited for the projection to become non-zero. The new initialization keeps
the initial mask close to the prior and makes query spatial gradients available
on the first optimizer step.

Activating that path exposed a second bug: invalid room memory tokens could
still influence mask logits through the recoverable attention floor. Invalid
room-token values are now zeroed before attention memory concatenation, while
the finite attention mask remains available for gradient recovery.

The adaptive-reference offset and context projections also use `std=0.001`
instead of exact zero. This makes sampled feature locations and the query update
differentiable on the first optimizer step rather than only after the two output
heads leave zero.

The same rule now applies to dynamic residual adapters:

- dynamic channel-query hyper output;
- spatial conv-adapter up projection;
- LoRA query, key/value, and output up projections;
- channel-family query deltas.

Previously each zero up/out projection allowed only that final layer to update
on the first step. Their down/source-conditioned paths now receive non-zero
gradients immediately.

Dynamic CAD stage refiners follow the same rule. Their output convolution,
spatial gate, stage-token gate, and route-budget output now use small non-zero
initialization, so room, wall, opening, and furniture stage trunks receive
first-step gradients instead of waiting for the final projection to move.

Furniture conditioning follows the same pattern. Stage film, room fusion, the
room-conditioned semantic query/room/gate projections, and the room-furniture
prior now use small non-zero residuals. This makes program order, room context,
and furniture semantic routing participate in the first optimizer step.

The remaining residual projection audit also activated:

- field evidence fusion;
- action-stage route budgets and gates, including their hidden trunks;
- program background, room structure adapter, and query-set global/route
  branches;
- source/cross-attention gates and all-channel reliability.

After excluding standard bias/shift/residual names, the fresh candidate scan has
19 multi-element zero parameters left. They are attention `in_proj_bias`
tensors and `program_router.class_bias`. These are ordinary zero biases and do
not block upstream feature trunks.

The active-query overlap objective was implemented in the shared mask loss but
was not exposed by predicted-control `instance_room_losses`, so predicted
control always used zero weight. It now flows through configuration, is logged
as `instance_room_query_separation`, and is enabled at `0.1` in the identity
candidate while remaining disabled by default.

The end-to-end binding now has one source of truth:
`typed_channel_instance_loss_kwargs`. It maps parsed candidate arguments to the
instance-loss call, including separation weight and class-agnostic mode.
Separation also consumes the authoritative count-selection gate instead of
class-logit presence alone. The identity candidate enables class-agnostic
duplicate suppression.

`room_mask_partition_loss` now consumes the same `sigmoid_existence` fusion path
as the final room readout whenever query presence is present. The previous
fallback `sigmoid` path could train a different query-to-partition mapping when
an explicit count selection was unavailable.

Query separation is also an atomic raw loss now. `instance_room` excludes the
separation term, the configured weight is applied exactly once, and the dynamic
balancer owns a separate `query_separation` scale. This prevents both hidden
weighting and duplicate counting.

The typed opening host gate now starts at `1e-3` instead of exact zero. This
keeps the adjustment near identity while connecting wall, junction, and room
boundary host evidence to the opening stage on the first step.

Validation now emits `query_separation_proxy` from the same count-selected gate
used in training. This closes the feedback loop: the proxy becomes zero when
fewer than two queries are active and remains positive for overlapping active
queries.

The proxy is registered as `RM-02.query-separation`, with minimize direction
against canonical Instance F1. It remains diagnostic until the standard
proxy-real consistency contract passes.

## Metric Ownership

| Weak canonical metric | Directly affected path |
| --- | --- |
| Room IoU | Program action token -> room-token class fusion; query-set route -> presence/count; field evidence -> room head. |
| Instance F1 | Query-set presence/count cold-start repair and room identity conditioning. |
| Node F1 | Room identity and presence feed graph conditioning and graph-node construction. |
| Edge overlap | Room identity feeds adjacency endpoints; no independent edge claim is made until matched evaluation. |
| GED | Downstream consequence of room count, type, node, and edge errors; this audit does not claim a GED gain. |
| Wall F1 | Not directly repaired by this zero-path audit; the known full-stage candidate-source blocker remains. |
| Opening F1 | Not directly repaired by this zero-path audit; vector-host training and interval evidence remain separate blockers. |

## Contract Evidence

The focused contract suite passes:

```text
42 passed, 1 existing checkpoint-compat warning
```

The full non-chip/non-NLP regression suite passes:

```text
1219 passed, 1 skipped
```

It verifies non-zero first-step gradients for:

- field evidence fusion;
- room structure adapter;
- room program class fusion;
- room query embeddings through the mask path;
- adaptive reference offsets and context updates;
- dynamic query hyper and family source gates;
- spatial adapter and LoRA low-rank input paths;
- CAD stage-refiner output, spatial/token gates, and route-budget trunks;
- furniture program film, room fusion, and room-conditioned semantic router;
- field/action/query residual projection trunks;
- invalid room-token invariance;
- action program route budget and gate;
- room query-set route budget;
- the core room instance, graph, furniture, and CAD action paths.

This is architecture and optimization-path evidence, not a validation or SOTA
result. A matched small-subset screen is still required before any metric claim.
