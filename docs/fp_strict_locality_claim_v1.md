# Strict-Locality Claim Decision v1

Status: current claim language updated 2026-11-02 from the per-operation
`residential_sota_main_v5` evidence. The full protocol authority is
`docs/fp_oracle_control_protocol_status_v1.md`.

## Decision

The requested “first” claim is not released. The current search did not identify
a prior floorplan method that reports the full FloorFlow combination, but the
search coverage is not yet strong enough for a priority assertion:

- OpenAlex and Semantic Scholar were rate-limited.
- DBLP returned an anti-bot page instead of query results.
- CVF/IEEE full-text proceedings were not exhaustively searched.
- Several current works use structure-preservation language without reporting
  exact outside raster equality.

The claim is therefore downgraded, not because the evidence is weak, but because
priority requires stronger search coverage.

## Released claim

> Under the control-gated local-delta protocol, FloorFlow provides an unusually
> complete auditable combination of edit quality, strict outside preservation,
> topology legality, structural consistency, and matched editing-aware
> comparison evidence on CubiCasa5K test503.

This claim is supported by:

- Delta-IoU `0.956703-0.999801`
- Delta-MSE `0.000019-0.000096`
- exact outside pixel, room, polygon, wall, opening, and edge content `1.0`
- FP-Eval validity `1.0`
- independent FEVP five-check validity `0.9124-0.9569`
- conditional watertightness `0.9579-0.9675`
- matching Raster2Seq baseline and an outside-frozen editing-aware wrapper
- oracle-control aligned Node F1 `0.982250`, edge overlap `0.929475`, and GED
  `0.869544`

| Operation | Delta-IoU | Delta-MSE |
|---|---:|---:|
| room type | 0.999199 | 0.000069 |
| room add | 0.999801 | 0.000030 |
| room remove | 0.999756 | 0.000019 |
| opening delete | 0.956703 | 0.000096 |
| furniture delete | 0.987812 | 0.000083 |

The published HouseMind-E anchors are Delta-IoU `0.608` and Delta-MSE `0.0019`.
The current FloorFlow values numerically exceed both anchors for every operation.
The comparison remains protocol-aligned rather than a same-dataset rank.

## Mandatory boundary

FloorFlow does not claim to beat HouseMind-E's complete natural-language editing
method and does not claim complete CAD-generation SOTA.

HouseMind-E reports on RPLAN natural-language edits. FloorFlow's main evidence
is CubiCasa control-gated editing. The numbers may be shown in a
protocol-coverage table as numeric metric context, but not ranked as a
same-benchmark result.

## Closest screened works

- **ChatHouseDiffusion** describes precise localized adjustments and provides
  public code, but the paper abstract and README do not report an exact
  outside-pixel/outside-label preservation protocol.
- **RenoLayout** emphasizes preserving immutable structural components, but its
  interface is design-point-guided renovation and it does not expose the same
  strict outside-raster metric in the screened material.
- **ReLayout** preserves unedited layout structure through relation graphs, but
  it is a generic design-layout editor rather than a floorplan CAD protocol.
- **HouseMind-E** supports localized natural-language editing, while the editing
  table omits strict outside exactness, validity, and conditional watertightness.
- **HypergraphFormer** improves editable representation for generated plans, but
  the screened material does not establish strict local editing of an existing
  raster.

## Conditions to unlock “first”

The phrase may be restored only after:

1. OpenAlex or an equivalent citation index is queried with an API key.
2. Semantic Scholar and DBLP coverage is completed or replaced by an equivalent
   venue-level search.
3. CVPR, ICCV, ECCV, AAAI, and SIGGRAPH-adjacent proceedings are screened.
4. Official code or papers are checked for exact outside-preservation metrics.
5. No prior method is found that satisfies all five dimensions.

## Reviewer-facing argument

The defensible argument is completeness under an explicit protocol: FloorFlow
reports edit quality, exact outside preservation, topology legality, structural
consistency, and matched editing-aware comparisons in one auditable setting.
The paper should invite comparison on those dimensions rather than claim global
superiority.

