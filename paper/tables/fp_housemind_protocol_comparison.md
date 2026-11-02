# HouseMind-E and FloorFlow Protocol Comparison

Status date: 2026-11-02. This table uses the current per-operation evidence in
`residential_sota_main_v5.md`.

This is a protocol-coverage comparison, not a same-dataset ranking. HouseMind-E
is evaluated on the 2,308-case HouseMind RPLAN edit split with natural-language
instructions. FloorFlow's main numbers are measured on CubiCasa5K test503 with
declared oracle controls and control-gated local deltas.

| Dimension | HouseMind-E | FloorFlow | Comparability |
|---|---|---|---|
| Edit input | Natural-language instruction | Control-gated local delta | Different input contracts |
| Evaluation data | 2,308 RPLAN cases | 503 CubiCasa cases, five seeds | Not directly comparable |
| Edit-detection quality | Delta-IoU 0.608; Delta-MSE 0.0019 | Delta-IoU 0.956703-0.999801; Delta-MSE 0.000019-0.000096 | Metric definitions aligned, data and task inputs differ; FloorFlow exceeds both numeric anchors |
| Final-layout pixel quality | Micro IoU 0.855; Macro IoU 0.823 | Separate CubiCasa target metrics; not ranked here | Not directly comparable |
| Node F1 | 0.998 | 0.982250 | Different data and aligned graph readout; FloorFlow is lower |
| Edge overlap | 0.934 | 0.929475 | Different data and aligned graph readout; FloorFlow is lower |
| GED | 0.467 | 0.869544 | Different data and graph readout; FloorFlow is worse |
| Outside pixel/room/polygon/wall/opening/edge exact | Not reported as editing metrics | 1.0 | FloorFlow-only strict-locality evidence |
| Validity | Not reported in the editing table | 1.0 | FloorFlow-only explicit metric |
| Conditional watertightness | Not reported in the editing table | 0.9579-0.9675 | FloorFlow-only explicit metric |
| Structure under strict locality | Not measured or reported | Oracle-control appendix | FloorFlow covers an additional audited dimension |
| Official inference pipeline | Official code marked “Code Soon” | Frozen experiment package supplied | Full reproduction unavailable |

HouseMind's editing prompt asks the model to preserve unrelated regions, but its
reported editing metrics do not include exact outside-pixel or outside-label
equality. FloorFlow therefore adds measured strict-local preservation; it does
not beat or supersede HouseMind-E's complete natural-language method.

The positive empirical result is the dual Delta metric. The aligned Node,
Edge, and GED values are below the HouseMind-E anchors and must be reported as
a limitation rather than hidden.

## Sources

- HouseMind paper: <https://arxiv.org/abs/2603.11640>, Table 3 and Appendix B.3.
- FloorFlow main table:
  [residential_sota_main_v5.md](residential_sota_main_v5.md).
- FloorFlow canonical structure:
  [residential_graph_protocol_dominance_v1.md](residential_graph_protocol_dominance_v1.md).
- HouseMind project page, “Code (Soon)”:
  <https://qinsz01.github.io/housemind/>.

