# CVPR Numerical Citation Policy v1

This policy applies to the abstract, introduction, experiments, captions, and
rebuttal material.

1. Every numerical result cited in the manuscript must resolve to a JSON or
   Markdown source file inside `outputs/cvpr_package_fp_v17.tar.gz`.
2. Historical reports `fp_cvpr_evidence_v6.json` through
   `fp_cvpr_evidence_v11.json` are not manuscript sources.
3. A number without a matching v17 artifact path must not enter the manuscript.
4. Captions and prose must retain the source boundary: CE2EPlan is a
   generation-only portability target and Raster2Seq is a matched
   reconstruction baseline.
5. Before submission, rerun the source audit and archive checksum verification.

Canonical source map:

- Main editing and locality numbers:
  `outputs/reports/fevp_fiveseed_stability_v2.json`
- Graph metrics and paired tests:
  `outputs/reports/fevp_graph_metrics_v2.json`
- Protocol attacks:
  `outputs/reports/fevp_attack_supplement_v1.json`
- External protocol boundaries:
  `outputs/reports/external_baseline_protocol_audit_v2.json`
- Final claim and reviewer boundary:
  `outputs/reports/fp_cvpr_evidence_v12.json`
