# Condition-to-floorplan objective

## What the previous run optimized

The condition-only run was not trained with multiflow's molecular loss
portfolio. It used a project-local sum:

```text
full 65-class weighted CE
+ 0.10 * foreground boundary focal/Dice over all 64 target channels
+ 0.10 * wall focal/Dice
+ 0.50 * visible-room CE/Dice
```

Checkpoint selection used `room_macro_iou`, while `best.pt` used full
validation CE. These are different functionals. The first-step cosine
scheduler also restarted with an almost-zero warmup LR when initializing from a
checkpoint.

The one-batch audit in
`outputs/diagnostics/condition_objective/objective_audit.json` measured:

| term | share of legacy scalar objective |
| --- | ---: |
| full 65-class CE | 56.2% |
| visible-room CE/Dice | 38.4% |
| foreground boundary | 3.0% |
| wall | 2.4% |

The full-class term was therefore the majority objective even though only room
IoU selected the checkpoint. Its "boundary" term was also not a room boundary:
it was a foreground boundary over all 64 channels.

## Current objective

The primary branch predicts a dense room partition over the complete interior
footprint:

```text
L = 1.00 * dense_room_partition(weighted CE + macro Dice)
  + 0.15 * soft_partition_boundary
  + 0.20 * opening_group(CE + Dice)
  + 0.20 * furniture_group(CE + Dice)
  + 0.10 * wall(BCE + Tversky)
```

Room semantics come from the partition head initialized with a semantic-head
prior. Walls, openings, and furniture are separate structured overlay heads.
The final drawing is assembled in the same order during validation, checkpoint
selection, and deployment:

```text
room partition -> furniture -> wall -> opening
```

The room partition target fills every pixel inside the footprint using the
nearest visible room label. Dense CE and macro Dice make the training objective
closer to the final room-rendering functional. The boundary term uses only room
class transitions; it does not average a foreground boundary over all classes.

## Checkpoint selection

`best_room_iou.pt` and `best.pt` are selected on the composed single-sample room
macro IoU. `best_balanced.pt` additionally uses opening F1, furniture IoU, wall
IoU, and a room-coverage shortfall penalty. No target-informed candidate
selection or best-of-N is used.

## Diagnostic contract

`scripts/audit_condition_objective.py` reports raw values, explicit weights,
signed/absolute scalar shares, per-term gradient norms, and cosine conflict with
the primary gradient. Gradient auditing defaults to one optimizer step and is
not a training-time performance claim.

`scripts/analyze_condition_failures.py` reports per-class IoU, worst cases,
component fragmentation, and a fresh-rasterization check against the cache. It
also saves worst-case prediction panels when given a checkpoint.

## Evidence before the next training run

Legacy logs labeled `train_iou` were misreported: the value was
`1 - room_soft_iou_loss`, not a pixel or room IoU. The corrected log field is
`train_room_soft_iou_loss`; true post-decoding room IoU must be computed
independently. On the retained checkpoint, true raw Room IoU was `0.1233` on a
64-case train slice and `0.0961` on val64, so the model was underfit rather than
showing the previously suspected train/test memorization gap.

- The cached target matched fresh SVG rasterization exactly on 10/10 checked
  test cases.
- Coverage was 0.983 while room macro IoU was 0.097. The contour was filled, but
  the room partition was wrong.
- Bedroom, LivingRoom, and Storage were the only room classes with meaningful
  mean IoU; Hall, Den, Kitchen, Closet, Dining, Office, Utility, and Sauna were
  effectively absent.
- Mean room component gap was 5.87, consistent with fragmented or merged room
  regions.
- The new objective gives the dense room partition 72.1% of scalar loss mass in
  the one-batch audit. Auxiliary wall gradients are the largest remaining term
  (30.1% of primary gradient norm) but are nearly orthogonal (cosine 0.04).

These diagnostics make loss and representation the current priorities. ODE
step count is not the explanation for the condition-only checkpoint because
that evaluation short-circuits to the semantic endpoint without an ODE solve.

## Reduced-scope lossfix check

The fixed objective was initialized from the existing condition checkpoint and
the partition head was initialized as an exact copy of its room logits.
Structured object heads start from cache-derived null priors. On a fixed
512-train / 96-val diagnostic:

| step | training partition loss | raw partition room IoU |
| --- | ---: | ---: |
| initialization | - | 0.0770 |
| epoch 1 | 2.3424 | 0.0844 |
| epoch 2 | 2.3137 | 0.0823 |
| epoch 3 | 2.3055 | 0.0892 |

The room branch is no longer best at the first epoch: it improves by about 16%
relative over initialization after three short epochs. The composed room IoU is
not monotonic yet because the object overlays are still immature; an early run
with object heads copied from an unrelated backbone reduced coverage to 0.37,
and subsequent runs still let wall predictions overwrite room pixels.

The object group heads now have learned null logits. Their background priors
come from the cache pixel frequencies (opening 0.79%, furniture 3.76%, wall
2.33%) instead of assuming a zero background logit. Training and probability
decoding now share that null class. These are reduced-scope diagnostics, not
publication metrics.

## Full-data result and isolation

The fixed dense-partition objective was trained on all 5,126 training floors.
Raw room IoU over six epochs was:

```text
0.1078, 0.1049, 0.1087, 0.1100, 0.1076, 0.1103
```

so checkpoint selection no longer latches onto epoch 0. In a controlled
1024-case run, adding object losses to the shared backbone reduced raw room IoU
from 0.0953 to 0.0765. The current object heads therefore consume detached room
features and are trained in a second stage with the room backbone frozen.

The 503 test channel/head metrics are opening F1 0.3523, furniture IoU 0.1451,
and wall IoU 0.3913. Exclusive single-raster composition reduces visible
furniture IoU to 0.0417. This is an output-composition problem, not a second
furniture model.

Additional negative controls:

- ResNet18-FPN direct contour segmentation reached only 0.0849 room IoU on the
  191-case validation subset, so increased encoder capacity was not adopted.
- A room-program presence head and a hierarchical group head both trained
  without exceeding the frozen dense-partition baseline and remain optional.
- Fine room macro IoU is 0.1166 while functional-group macro IoU is 0.1476;
  many residual errors are near-neighbor room-type confusions.
