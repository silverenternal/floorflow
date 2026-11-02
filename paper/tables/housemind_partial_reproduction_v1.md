# HouseMind Partial Reproduction

| Arm | Delta-IoU | Delta-MSE | Node F1 | Edge overlap | GED | Role |
|---|---:|---:|---:|---:|---:|---|
| HouseMind-E published | 0.608 | 0.0019 | 0.998 | 0.934 | 0.467 | reported reference ceiling |
| HouseMind public weights | N/A | N/A | N/A | N/A | N/A | partial: 16 token-generation prompts, 1560 room tokens |
| FloorFlow strict-locality edit | 0.9567-0.9998 | 0.000019-0.000096 | aligned protocol | aligned protocol | aligned protocol | different task/protocol |

Official image tokenization, sequence serialization, constrained decoding, VQ-VAE decoding, and post-processing remain unavailable. No direct HouseMind rank is claimed.
