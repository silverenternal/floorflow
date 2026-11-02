# HouseMind-Style Structural Alignment on CubiCasa

HouseMind-E reference (RPLAN, natural language): Node F1 0.998, Edge overlap 0.934, GED 0.467. These values are not a same-dataset rank.

| Seed | Before GT NodeF1/Edge/GED | Before FloorFlow | After FloorFlow |
|---:|---:|---:|---:|
| 19 | 0.9643 / 0.8722 / 1.5805 | 0.9643 / 0.8672 / 1.6551 | 0.9828 / 0.9316 / 0.8447 |
| 20 | 0.9643 / 0.8722 / 1.5805 | 0.9643 / 0.8682 / 1.6426 | 0.9828 / 0.9325 / 0.8245 |
| 21 | 0.9643 / 0.8722 / 1.5805 | 0.9627 / 0.8614 / 1.7576 | 0.9802 / 0.9228 / 0.9835 |
| 22 | 0.9643 / 0.8722 / 1.5805 | 0.9643 / 0.8668 / 1.6595 | 0.9828 / 0.9316 / 0.8326 |
| 23 | 0.9643 / 0.8722 / 1.5805 | 0.9643 / 0.8662 / 1.6700 | 0.9827 / 0.9289 / 0.8624 |

The HouseMind-style view collapses instances to unique room types and type-pair edges. The canonical view performs one-to-one room-instance IoU matching with explicit unit costs for edge edits.

Both use oracle controls and CubiCasa test503; neither is a direct HouseMind-E RPLAN leaderboard reproduction.
