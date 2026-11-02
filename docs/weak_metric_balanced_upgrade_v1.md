# Weak-Metric Balanced Upgrade v1

Date: 2026-11-01

The predicted-stage budget in this document is superseded by
`docs/predicted_stage_mita_graph_v2.md`; the room-head audit remains current.

## Audit

The model is not uniformly underparameterized. Its capacity is concentrated in
the dense encoder while several weak metrics are produced by very small heads.

| System | Before | Problem |
|---|---:|---|
| Room encoder | 12.39M parameters, 94.0% | dominant capacity |
| Room query decoder | 0.69M, 5.2% | underweighted identity readout |
| Room count head | 0.10M, 0.8% | small global set head |
| Wall stage head | one 1x1 conv | no boundary/owner dynamics |
| Opening stage head | one 1x1 conv | weak host-wall conditioning |
| Furniture stage head | one 1x1 conv | weak room containment conditioning |
| Typed channel screen | 676k parameters, 2,048 exposures | underbudgeted |

The old structure bridge also used only 774k parameters, 3 full epochs and
about 15.4k sample exposures. That is not a stable budget for wall, opening and
given the width and stroke relations. A larger budget is only justified after
the page-floor/protocol distribution is aligned.

## Parameter Rebalancing

### Room readout candidate

The alternative room configuration increases:

- hidden width: 128 -> 192
- queries: 24 -> 32
- decoder layers: 3 -> 5

The query decoder share rises from 5.2% to 15.9%, while the count head rises
from 0.8% to 2.9% through query-set self-attention and a wider global context.
The encoder is not expanded. This targets room identity and cardinality without
adding more dense capacity to an already dominant encoder.

These values are now the defaults for new raster-instance runs. Old checkpoint
configs that omit the new fields still reconstruct the previous 128/3/24 model
with a 64-wide count head.

### Predicted stage heads

The generic one-line stage classifier is replaced by a dedicated dynamic head
per room, wall, opening and furniture layer:

- field projection;
- prior projection;
- stage-token FiLM;
- two local convolutional residual blocks;
- image-conditioned per-pixel gate;
- stage-token gate;
- stage output residual.

Wall heads are conditioned on room probability, opening heads on wall
probability, and furniture heads on room probability.

### Dynamic stage graph

The CAD action graph now adds a per-image low-rank link bias between a stage and
its declared predecessors. The family prior and learnable global residual remain
as stable priors; the new term adapts routing to the current predicted graph.

## Budget

| Component | Screen | Promotion | Confirmation |
|---|---:|---:|---:|
| Room readout | 512 x 12 | 1,024 x 16 | 5,122 x 16 |
| Predicted channels | 512 x 12 | 1,024 x 16 | 5,126 x 16 |
| Structure bridge | 512 x 12 | 1,024 x 16 | 5,126 x 12 |

The predicted-channel model is approximately 2.23M parameters. Each dynamic
stage head is approximately 242k parameters, replacing the prior small 1x1
stage classifiers.

## Attribution Plan

The upgrades must be evaluated separately before combining:

1. room decoder capacity only;
2. evidence context only;
3. dynamic channel graph only;
4. dynamic stage heads only;
5. dynamic stage graph only;
6. full combination at matched budget;
7. parameter-matched random-capacity control.

Every arm reports room, wall, opening, furniture, Node, Edge, GED, validity,
watertightness and outside-footprint metrics.

## Boundary

No weak metric is claimed improved until the matched runs finish. This pass
changes capacity, routing and budget contracts; it does not add a post-hoc
metric rescue rule.
