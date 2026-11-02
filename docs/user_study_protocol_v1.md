# Style-conditioned floorplan design: frozen user-study protocol

## Research questions

1. Do style-conditioned generations receive higher overall preference than
   contour-only generations?
2. Do raters judge style-conditioned generations as better matched to the
   requested style?
3. Does style control preserve perceived legality and room coverage?

## Conditions

The pilot compares three conditions on identical test contours:

- `style_conditioned`: frozen checkpoint, user-selected style, calibrated gate.
- `no_style`: same checkpoint with the style input disabled.
- `retrieval`: train-only, style-constrained nearest-contour retrieval baseline.

All conditions use one sample per contour. Best-of-N and target-derived controls
are prohibited.

## Participants

- Pilot: 6 architecture/interior-design professionals and 12 general users.
- Full study: 12 professionals and 24 general users if the pilot has enough
  usable responses.
- Exclusion: no responses for a case, failed attention checks, or participation
  in dataset annotation.

## Presentation and blinding

- Randomize condition order independently for every participant and case.
- Hide model names, style ids, and generation method.
- Show the contour, an explicit style token, four train-only style reference
  examples, and the generated plan without GT.
- Hold image resolution, margins, fonts, and color palette constant.
- Do not show a best sample selected by model scores.

## Primary measures

- Pairwise overall preference.
- Style fit on a 7-point Likert scale.
- Perceived legality: binary pass/fail with reason labels.
- Preferred plan among all three conditions.

## Secondary measures

- Confidence, drawing readability, perceived room coverage, and whether the
  opening/furniture relationships appear plausible.

## Statistics

- Report all conditions and all cases; no post-hoc condition removal.
- Use a mixed-effects logistic model with random intercepts for participant and
  case for pairwise preference.
- Use mixed-effects ordinal/logistic models for Likert style fit and legality.
- Report bootstrap 95% CIs for preference rates.
- Report inter-rater agreement with Krippendorff's alpha.
- The pilot estimates variance; it is not a confirmation study.

## Integrity gates

- Freeze participant instructions, case ids, randomization seed, and survey
  schema before recruitment.
- Record missingness and exclusions before analysis.
- Keep raw participant identities outside the repository.
- Never include reviewer or user preferences as a model-selection score.

## Current execution status

- Pack v2: `outputs/reports/user_study_case_pack_v2.json`.
- Offline survey: `outputs/user_study_pilot_v2/survey.html`.
- Blinded answer key: `outputs/user_study_pilot_v2/answer_key.json`.
- Ethics status: PENDING institutional review.
- Recruitment and response submission are disabled until ethics status is
  recorded as READY.

