# Opening Candidate Ranker v2

Date: 2026-11-01

## Bottleneck

The multiscale detector improved full-val491 Opening F1 from `0.08702` to
`0.26639`, but its oracle-host ceiling is still low. The `0.15` detector
threshold produces `11.97` candidates per case, while only `4.14` match a
target. Candidate verification is therefore the next independent atom.

## Method

The candidate extractor reads 1D intervals from independent foreground
probabilities. It uses family logits only for the optional family classifier.
Each candidate is represented by 21 target-free features spanning detector
score, foreground profile, family margin, interval geometry, host-wall length
and type, orientation, candidate density, and neighboring spacing.

Two deterministic histogram-gradient-boosting models predict candidate
presence and door/window family. A single threshold and family mode are selected
on `val[0:96]`. `val[96:288]` is a disjoint holdout. At inference no target is
used.

## Result

On the disjoint oracle-host holdout, presence-only ranking raises vector F1 from
`0.304204` to `0.316145`. The paired gain is `+0.011942`, 95% CI
`[0.006454, 0.017521]`, rank-biserial `0.566018`, Cliff's delta `0.4375`, and
one-sided Wilcoxon `p=6.09e-8`. Geometry retention is `98.1%`; the family
consistency rate is unchanged.

The family classifier is not promoted: it lowers matched-window recovery.
Presence filtering with detector-assigned family is the selected arm.

## Boundary

Oracle-host vector F1 remains `0.316`, below the absolute `0.35` gate. The arm
passes the full predicted-wall consumer: Opening F1 rises from `0.266386` to
`0.282302`, with paired `+0.015916`, 95% CI `[0.008116, 0.023715]`, and
Wilcoxon `p=2.57e-5`. Room, wall, node, edge, deterministic GED, validity,
watertightness, and outside-footprint remain unchanged on all 491 cases.

The ranker is now a promoted relative mechanism. Absolute Opening F1 remains
below `0.35`, so it does not support an opening or full-CAD SOTA claim.
