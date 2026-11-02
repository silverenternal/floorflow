# Editing-task Audit Evidence Map

| Method | Role | Outside exact | Violation rate | Negative controls | Adversarial checks | Paired stats |
|---|---|---|---|---|---|---|
| FEVP + FloorFlow | evaluated | ✓ | ✓ | ✓ | ✓ | ✓ |
| HouseMind-E | editing context | — | — | — | — | — |
| HouseMind-O | unified editing context | — | — | — | — | — |
| HouseMind-G | generation only | n/a | n/a | n/a | n/a | n/a |
| CE2EPlan | generation context | — | — | — | — | — |
| Raster2Seq | reconstruction context | — | — | — | — | — |
| Raster2Seq outside-frozen wrapper | editing wrapper | pixel only | — | — | — | — |

— means the method does not report the corresponding audit evidence; it does not mean the method lacks the property or performs worse.
