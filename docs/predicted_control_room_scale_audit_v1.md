# Predicted-Control Room Scale Audit v1

## Evidence

The authoritative full validation has Room support IoU `0.7743` but semantic
Room macro IoU `0.3276` and Instance F1 `0.2818`. The decoder therefore finds
most room foreground while assigning it to the wrong room instance, semantic
class, or spatial extent.

The matched 64/32 seed19 logs expose a scale failure that was previously
diagnostic-only:

| Arm | Epoch | Matched predicted area | Target area |
| --- | ---: | ---: | ---: |
| Identity control | 0 | 0.003650 | 0.027531 |
| Identity control | 1 | 0.011754 | 0.026963 |
| Graph joint | 0 | 0.045550 | 0.027531 |
| Graph joint | 1 | 0.041914 | 0.026963 |

Control masks are heavily under-scaled. The unconstrained graph bundle flips to
over-scaling, so its room regression is not evidence that graph conditioning is
useless; it is evidence that owner redistribution needs both area preservation
and explicit scale supervision.

## Code Repair

`floorflow.heads.room_mask_area` adds a bounded symmetric relative area error
over already matched queries:

```text
area_error(query) = |pred_area - target_area| / (pred_area + target_area + eps)
```

The term has a gradient even for a collapsed mask, saturates near one instead of
diverging, and treats over- and under-segmentation symmetrically. It enters
through `room_instance_mask_area_weight`, is logged as `instance_room_area`, and
is a separate `room_area` term in the dynamic balancer so it is neither hidden
inside Dice nor counted twice.

The default remains disabled for historical checkpoints.
`configs/predicted_channel_v6_area_fidelity.json` enables `0.2` on top of the
area-preserving graph candidate. Its matched control is the same configuration
with that weight set to zero.

`configs/predicted_channel_v6_area_fidelity_only.json` isolates the loss against
the identity control. A flattened configuration diff confirms that each pair
changes only the area weight and descriptive metadata, so a later matched screen
will not confound this loss with another architecture flag.

## Gate

A matched screen must show:

1. lower absolute predicted-target area error;
2. no material Room or Instance regression on any seed;
3. improved Room IoU or Instance F1 with a same-case confidence interval;
4. unchanged count selection and strict-locality contracts.

This is an architecture repair, not a metric result.
