# Chip Atom Heads OOF Screen v1

Date: 2026-11-02

## Protocol

Six independent ExtraTrees atom heads were trained on the 408 training cases,
with five case-disjoint folds and four candidates per case. The 92 development
cases were not used for model selection. All labels are legalized candidate
metrics from `chip_s3_candidate_legalized_labels_v3_500_v1.json`.

Artifacts:

- `scripts/train_chip_atom_heads.py`
- `src/floorflow/chip_flow/atom_score_heads.py`
- `outputs/reports/chip_atom_heads_oof_v1.json`
- `outputs/train/chip_atom_heads_oof_v1/atom_heads.pkl`

## OOF Result

| Atom | Mean Spearman | Pairwise accuracy | Oracle top-1 | Mean rank regret |
|---|---:|---:|---:|---:|
| grouping | 0.339 | 0.686 | 0.311 | 0.355 |
| area/rho | 0.550 | 0.746 | 0.549 | 0.020 |
| b2b | 0.583 | 0.766 | 0.539 | 0.008 |
| boundary | 0.256 | 0.649 | 0.216 | 0.203 |
| p2b | 0.309 | 0.636 | 0.400 | 0.047 |
| legality/tail | 0.379 | 0.671 | 0.407 | 0.112 |

## Interpretation

Area/rho and b2b are the strongest learnable atom signals. Grouping,
legality/tail and p2b are moderate. Boundary is the weakest independent head,
consistent with its strong mean gain but unstable per-case ordering.

The atom heads are target-free at inference and consume only the existing
14-dimensional candidate features plus masked atom context. They are not a new
backbone and do not require rerunning flow sampling during head training.

The OOF predictions are the only inputs permitted to train the typed
cross-attention aggregator. A selector or aggregator cannot exceed the candidate
pool oracle, so the next screen must compare pool-oracle changes before adding
another generator.

## Typed Aggregator Screen

The typed aggregator was trained only on the 408 OOF cases and evaluated on the
92-case development split.

- Hidden 32, 200 steps: JSR `28.3%`, CSJ `8.7%`; rejected for overfitting.
- Hidden 8, 50 steps: JSR `79.35%`, CSJ `50.0%`, versus the fixed selector's
  `79.35% / 51.09%`; this regularized arm preserves primary behavior but does not
  improve coverage.

The aggregator remains default-off. These results mean the current atom evidence
can explain candidate quality, but aggregation over the existing four candidates
cannot create the missing constraint-safe wins. Candidate generation is the next
hard gate.
