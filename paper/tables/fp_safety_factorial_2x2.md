# Safety Mechanism Factorial (2x2)

| Arm | Outside exact | Room exact | Delta-IoU | Delta-MSE | Validity | Watertight | Wall breaks |
|---|---:|---:|---:|---:|---:|---:|---:|
| control_on_guard_on | 1.000000 | 1.000000 | 0.987754 | 0.000168190 | 1.000000 | 0.964502 | 0.000 |
| control_on_guard_off | 1.000000 | 1.000000 | 0.987754 | 0.000168190 | 1.000000 | 0.962888 | 0.000 |
| control_off_guard_on | 0.989987 | 0.987453 | 0.866409 | 0.010374545 | 1.000000 | 0.964502 | 0.000 |
| control_off_guard_off | 0.989987 | 0.987453 | 0.866409 | 0.010374545 | 1.000000 | 0.963695 | 0.000 |

The four arms share the same rows, checkpoints, control masks, and scorer.
Control off/guard on uses raw visible output with guarded topology; control on/guard off uses gated visible and unguarded topology.
