# Floorplan Submission Review v1

Status: review of the current four-page candidate draft.

## Editorial Decision

`major_revision_before_submission`

The draft now has a defensible research question, executable evidence, figures,
tables, and an explicit boundary. It is not submission-ready because the
predicted model and semantic audit are not frozen and no fresh held-out result
exists.

## Editor

The paper fits the vision and design-system audience. Its central thesis,
separating edit reliability from reconstruction fidelity, is clearer than a
single leaderboard claim. The strongest material is the combination of exact
locality, structural plausibility, and an honest semantic boundary.

The main risk is that the result may read as two unrelated regimes. The final
revision must explain why the old-control editing protocol connects them.

## Methodology Reviewer

Strengths:

- explicit zero-tolerance outside domain;
- separate raw and functional watertightness;
- separate FP-Eval and FEVP validity;
- deterministic operation-success predicates;
- source hash and artifact integrity checks;
- corrected split metadata for future evaluations.

Required revisions:

- rerun semantic audit after model freeze;
- use a fresh unopened split;
- report source-present and source-drift strata;
- clarify that repeated seeds are reproducibility evidence on deterministic
  oracle rows;
- add uncertainty for semantic success across cases.

## Domain Reviewer

The old CAD/control state is a legitimate editing input and should remain in the
method rather than being treated as leakage. The paper should use language that
matches engineering practice: an operator modifies an existing drawing from its
prior state.

The contribution is stronger as a protocol and reliability study than as a
generic floorplan generator.

## Perspective Reviewer

The practical benefit is conservative local editing with exact outside
preservation and safe refusal. The final paper should quantify the cost of
abstention and report supported-operation coverage, not only success rates.

## Devil's Advocate

Strongest counter-argument:

> The oracle result is target-derived, exact outside preservation follows from
> local composition, and the predicted setting lacks both high semantic success
> and a fresh held-out result.

Current defense:

- no-gate leakage and matched controls show the audit is non-constant;
- target-free structural plausibility is high;
- operation-success failures are disclosed rather than relabeled;
- the test-development metadata error is corrected.

This defense is credible but incomplete until a frozen predicted result is
available.

## Revision Roadmap

1. Freeze the improved predicted model and source contract.
2. Rerun operation semantic success on validation.
3. Freeze thresholds and evaluator hashes.
4. Open one fresh held-out split.
5. Replace candidate predicted columns with frozen results.
6. Recompile PDF and rerun anonymity, figure, claim, and artifact audits.

## Submission Decision

Do not submit the current draft as final. The scientific framing is viable; the
missing evidence is a frozen deployable result, not another round of document
polish.
