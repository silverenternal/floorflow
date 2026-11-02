# Distribution CFM Structure Fusion v2

Date: 2026-11-01

## Audit Response

The CFM remains a time-conditioned U-Net with coupled pixel and semantic
element distributions. The previous weaknesses were representation-level:

1. contour condition features were recomputed at every NFE;
2. room/wall/opening/furniture streams shared the same entrance tokens;
3. structural evidence entered only at the bottleneck;
4. pixel/element states used a convex soft blend;
5. no executable representation audit existed.

All five are now addressed.

## Static Condition Cache

`ConditionEncoder` can exclude time from contour processing. The condition
pyramid is independent of `x_t` and is computed once per sample before ODE
integration. Each NFE still computes the current-time U-Net representation.

For four Heun steps and eight NFE, condition-encoder reuse is `7/8 = 0.875`.
The shape descriptor is now a vectorized moment computation rather than a
per-sample nonzero/sort/FFT loop.

## Typed Family Streams

The room, wall, opening, furniture, and condition streams are no longer copied
from one hidden tensor. Family streams aggregate their own state channels and
condition evidence through separate projection paths. The velocity-scale
mixing also uses independent family projections.

## Multi-Scale Structure Injection

A typed structural token is built from the current state families and envelope.
Identity-initialized FiLM injects that token before every U-Net encoder level,
not only at the bottleneck.

## Residual CFM Bridge

The structured bridge supports a zero-initialized additive residual:

```text
state = pixel_state + tanh(channel_residual) * element_probability
```

Old channel checkpoints retain the original convex blend unless residual mode is
selected.

## Representation Audit

Run:

```bash
uv run python scripts/audit_distribution_cfm_representation.py \
  --checkpoint outputs/train/distribution_joint_full_v8_budgetfix/best.pt
```

The latest CPU audit records:

- condition cache reuse ratio `0.875`;
- family pairwise distinctness around `0.84-0.91`;
- zero CFM state outside the support;
- residual bridge mode;
- static condition, typed streams, and all-scale injection enabled.

## CPU Verification

The audit script completed on CPU. It uses an existing checkpoint only to
instantiate the representation; the newly attached structure modules are
untrained. The report is an architecture audit and makes no generation-quality
claim.
