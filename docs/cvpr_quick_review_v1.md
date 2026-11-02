# CVPR quick review v1

Status: historical review of the pre-protocol style-generation package. It is
superseded for submission positioning by
`docs/cvpr_oracle_protocol_presubmission_review_v2.md`.

## Material Passport

- Artifact: `docs/cvpr_evidence_pack_v1.md`
- Review mode: quick pre-submission EIC assessment
- Read-only status: no evidence or manuscript modified by the review
- Review date: 2026-10-24

## Field analysis

| Dimension | Assessment |
|---|---|
| Primary discipline | Computer vision and generative modeling |
| Secondary disciplines | CAD/floorplan generation and architectural computing |
| Research paradigm | Quantitative machine-learning experiments |
| Methodology | Conditional flow matching, distributional and CAD evaluation, ablations |
| Target tier | CVPR main conference |
| Maturity | Evidence pack, not yet a manuscript; pre-submission major revision |

## Reviewer configuration

The quick panel is an EIC for generative vision and design systems. The EIC's
focus is contribution framing, CVPR readership interest, and whether the
evidence supports the claimed task.

## EIC review

### Overall recommendation

**Major Revision before submission.** Not ready to submit, but the new scope is
substantially more defensible than the rejected contour-only exact-reconstruction
contract.

### Confidence

4/5 for publication positioning; 3/5 for manuscript quality because no full
manuscript exists yet.

### Summary assessment

The evidence pack proposes contour-conditioned, style-controllable constrained
floorplan generation with a shared MiTA-CFM backbone. The strongest results are
hard legality: test503 has zero footprint violations, room coverage 0.9983,
opening-wall alignment 0.9789, furniture-room containment 0.9996, and coverage
is nonzero for every evaluated style. The style bank is train-only and the
deployment style interface is explicit.

Official external assets and native/oracle-program baseline tables are now
present, and a deterministic distribution-overlay decoder raises all-style
CubiCasa test503 Room IoU from 0.10316 to 0.12383 while preserving zero
footprint violations. A second training seed reproduces the overlay val64 result
within 0.0001. The remaining major gaps are exact Room IoU, vector geometry,
coverage and watertightness; no human evidence is in scope.

### Strengths

1. **Legal outputs are strong and measured at scale.** The test503 report
   measures `outside_footprint_pixels=0`, room coverage `0.99826`, and
   opening/furniture relational scores above `0.97/0.99`.
2. **Style is a real deployment input, not hidden target leakage.** Both style
   banks are train-only and val-calibrated; cross-domain direct transfer is
   retained as a negative baseline.
3. **The task definition is honest.** Reference IoU is explicitly secondary,
   and the previous non-identifiability result is preserved rather than hidden.
4. **The system has a reproducible evidence pack.** The manifest hashes splits,
   checkpoints, style banks, seeds and gate metadata.

### Weaknesses

1. **External comparison is partly oracle-program.** House-GAN++ has a CubiCasa
   val64 oracle-program table and MaskPLAN has a native RPLAN table. A fully
   contour-only external comparison remains difficult and must be scoped
   carefully.
2. **CAD geometry remains weak.** Opening F1, room polygon F1, coverage and
   watertightness fail the hard CAD gate despite strong containment.
3. **Novelty needs a sharper statement.** Multi-channel CFM + hierarchical
   attention + style adapters is promising, but reviewers will ask what
   mechanism is uniquely responsible for the gain. The ablation matrix must
   isolate style token pathways and hierarchical channel coupling.
4. **Coverage varies across styles and datasets.** CubiCasa test coverage is
   strong, while skv4 style coverage is `0.271–0.532`; the paper must explain
   why coverage differs and whether it is a dataset or model issue.
5. **Conditional watertight is now measured but robustness is untested.**
   Test503 conditional watertight is `0.97973`, while unconditional is
   `0.91849`; a second seed is still required.

### Questions for authors

1. Which external baseline is most comparable under the same style-free
   condition, and can its official checkpoint be reproduced?
2. Does style control improve blinded human preference over no-style and
   retrieval baselines?
3. Which component causes the style separation: feature adapter, direct
   pixel/element bias, or velocity adapter?
4. Does the method preserve legality under a second seed and a held-out style
   bank?
5. Is conditional watertight stable across a second seed and domain?

## Editorial disposition

Do not submit yet. The minimum package is:

1. official or faithfully reproduced external baseline table;
2. target-free CAD gate with acceptable vector, coverage and topology metrics;
3. component-isolation ablation for the three style pathways and hierarchy;
4. second-seed robustness;
5. second-domain conditional watertight replication.

After these, the paper has a plausible CVPR submission story, but acceptance
cannot be guaranteed.

