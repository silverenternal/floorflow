# RPLAN Generator Scaling v2

## Result

The 4096 train / 512 disjoint-validation screen did not reach the Room IoU
`0.55` promotion gate. It did produce a real data-scaling gain and a useful
diagnosis.

On the same 256 validation cases:

| Train cases | Room IoU | Instance F1 | Type accuracy | Count MAE | Mask IoU | Macro IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.4773 | 0.2389 | 0.5806 | 1.195 | 0.5610 | 0.3284 |
| 4096 | **0.5163** | **0.2729** | 0.5557 | **1.078** | **0.5824** | **0.3611** |

The 4096 model improves Room IoU by `+0.0389`, instance F1 by `+0.0341`, mask
IoU by `+0.0214`, and count MAE by `-0.1172`. Type accuracy drops by `0.0249`,
so larger data alone does not solve semantic identity.

## Disjoint Validation

The checkpoint selected on validation offset 256 with 512 cases reaches:

- Room IoU `0.4908`;
- instance F1 `0.2719`;
- type accuracy `0.5580`;
- mask IoU `0.5878`;
- matched class accuracy `0.5807`;
- rendered room macro IoU `0.3655`;
- exact room count `0.4941`;
- count MAE `0.6133`.

Count errors are not the dominant factor: under-count rate is `10.7%`,
over-count rate is `39.8%`, but exact count is already `49.4%`. The larger loss
comes from mask/instance separation and semantic/geometry matching.

## Diagnosis

The current vector decoder is close to the identity-mask ceiling:

- mask IoU is `0.588`;
- rendered macro IoU is `0.365`;
- instance F1 is only `0.272`.

Therefore, additional polygon capacity or longer training on the same signal is
not the right next move. The next matched experiment should target:

1. query-mask separation and duplicate suppression;
2. class-balanced identity supervision;
3. class-conditioned mask assignment;
4. count calibration after those identity changes.

All ablations must use the same 1024/256 screen, seed, checkpoint score, and CAD
decoder. RPLAN test and CE2EPlan test outputs remain closed.
