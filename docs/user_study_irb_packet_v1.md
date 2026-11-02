# User-study ethics submission packet

Status: **TEMPLATE — NEEDS PRINCIPAL INVESTIGATOR AND INSTITUTION INPUT**

This packet does not constitute ethics approval and must not be used to begin
recruitment until the responsible institution records approval or exemption.

## Study identification

- Working title: Style-controlled constrained floorplan generation
- Principal investigator: `[REQUIRED]`
- Institution / department: `[REQUIRED]`
- Ethics contact and submission ID: `[REQUIRED AFTER SUBMISSION]`
- Funding source: `[REQUIRED OR "NONE"]`

## Purpose

Estimate whether explicit style conditioning improves perceived style match and
overall preference while preserving perceived architectural legality. The study
also estimates response variance for a later full study.

## Design and procedures

- Within-participant, blinded comparison of three anonymous conditions.
- Frozen pack: 32 CubiCasa test contours.
- Conditions: style-conditioned, no-style, and train-only retrieval.
- Each case shows the contour and four train-only style-reference examples.
- Every participant evaluates all 32 cases with per-participant randomized A/B/C
  slot order.
- Expected duration: 30–45 minutes. Participants may stop at any time.
- No target/reference floorplan is shown.

## Participants

- Pilot target: 6 architecture/interior-design professionals and 12 general
  users.
- Adults aged 18 or older.
- Exclusion: a person who contributed to the annotations, model development, or
  study design; no usable answer to the attention check.
- Recruitment method: `[REQUIRED: institutional channels and invitation text]`
- Compensation: `[REQUIRED OR "NONE"]`

## Risks and benefits

- Physical risk: none expected.
- Psychological or social risk: minimal; tasks involve design-quality judgments,
  not personal or sensitive topics.
- Benefit: no direct personal benefit; contribution to research on controllable
  floorplan generation.
- Participants may skip or stop the survey without penalty.

## Consent process

- Digital information sheet followed by an explicit click-to-consent checkbox.
- Consent must be recorded in the response JSON before any case data are saved.
- Participants can withdraw by closing the survey without downloading a response.
- Because responses are anonymous, withdrawal after an anonymized response has
  been returned may not be technically possible; this limitation must be stated.

## Data management

- Collected fields: participant role, experience years, blinded ratings, optional
  comment, timestamps, and anonymous download identifier.
- Do not collect names, email addresses, IP addresses, or institution names in the
  response file.
- Raw response files will be stored at: `[REQUIRED SECURE LOCATION]`
- Access: `[REQUIRED: named research team roles]`
- Retention period and deletion date: `[REQUIRED]`
- Transfer: `[REQUIRED: encrypted institutional transfer method]`
- Local law/compliance basis: `[REQUIRED]`

## Expected analysis

- Descriptive condition-level style-fit, legality, confidence, and preference
  rates.
- Participant-level bootstrap 95% CIs.
- Mixed-effects models only after the pilot supports a larger study.
- No response is used for model selection or training.

## Ethics checklist

| Category | Current packet status |
|---|---|
| Consent pathway | Drafted; PI/institution review required |
| Privacy and anonymization | Draft protocol; secure storage TBD |
| Risk assessment | Minimal-risk draft |
| Vulnerable populations | Adults only; no planned vulnerable population |
| Institutional approval | **PENDING** |
| Data retention and transfer | **REQUIRED** |
| Recruitment materials | **REQUIRED** |

## Required institutional decision

Record one of: `APPROVED`, `EXEMPT`, or `NOT_APPROVED`, together with the
approval/exemption ID, date, and responsible contact. Only `APPROVED` or
`EXEMPT` can change the survey status from PENDING to READY.
