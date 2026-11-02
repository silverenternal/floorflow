# FloorPlan Submission Figure Captions v1

## Protocol Overview

The control protocol separates the old CAD state, a declared typed request, and
the resulting local edit. Supported operations execute locally; unsupported or
ambiguous operations abstain. Every accepted output is checked for exact
outside preservation, topology validity, and structural plausibility.

## Predicted-Control Evidence

Safety, structural plausibility, and semantic operation realization are separate
axes. The safety contract is exact, and predicted-control outputs are
structurally plausible. Semantic success remains mixed in the preliminary
validation audit, especially for room removal and furniture deletion. The figure
is not a held-out test result.

## Failure Taxonomy

Supported operations are decomposed into semantic successes and failures with
the actual denominator shown for every category. The retained failures prevent
abstention or exact locality from being reported as evidence of semantic edit
correctness.
