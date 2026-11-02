# Room-Deletion Diagnosis

| Metric | Room deletion |
|---|---:|
| Delta-IoU | 0.994120 |
| Delta-MSE | 0.000605869 |
| Node F1 | 0.864368 |
| Edge F1 | 0.717480 |
| GED | 5.418000 |
| GED 95% Student-t CI | [5.293233, 5.542767] |
| GED cases | 2500 |
| Outside exact | 1.000000 |
| Locality violation | 0.000000000 |

| Tail metric | Value |
|---|---:|
| Overall GED P90 threshold | 7.000000 |
| Room-removal share of top-decile GED | 0.805164 |

Room deletion is the principal graph-distance difficulty: GED 5.418, Node F1 0.864, and Edge F1 0.717. It is not a locality failure: outside exactness remains 1.0 and violation rate remains zero on the same subset. FEVP therefore separates outside leakage from topological uncertainty intrinsic to deletion.

This diagnosis explains a hard mode; it does not claim room deletion is solved or that GED is comparable across evaluators.
The 135-case figure belongs to the 4+ room edit-strength bin, not room deletion.
