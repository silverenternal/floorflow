# Compact FEVP Attack Sensitivity

| Attack | Expected behavior | Baseline | Attacked | Observed |
|---|---|---:|---:|---|
| Move an unedited room polygon | Outside room-level exactness decreases | 1.000000 | 0.000000 | pass |
| Inject a 1-pixel hole | Watertight hole count increases | 0.000000 | 1.000000 | pass |
| Delete one adjacency edge | Edge F1 decreases | 1.000000 | 0.000000 | pass |
| Change one room label | Node F1 decreases | 1.000000 | 0.500000 | pass |
| Overlap two rooms | Validity room-overlap check decreases | 1.000000 | 0.000000 | pass |

These attacks validate evaluator sensitivity; they are not used to increase the primary method score.
