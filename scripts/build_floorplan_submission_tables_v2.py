#!/usr/bin/env python3
"""Generate the four scoped CVPR tables from registry-bound evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/floorplan_submission_source_registry_v2.json"
ORACLE_STATS = ROOT / "outputs/reports/oracle_submission_statistics_v2.json"
OPERATIONS = ("room_type", "room_add", "room_remove", "opening_delete", "furniture_delete")
LABELS = {
    "room_type": "Room type",
    "room_add": "Room add",
    "room_remove": "Room remove",
    "opening_delete": "Opening delete",
    "furniture_delete": "Furniture delete",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(value: Any) -> float:
    return float(value["mean"] if isinstance(value, dict) else value)


def _f(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _write(stem: str, markdown: str, latex: str) -> None:
    root = ROOT / "paper/tables"
    (root / f"{stem}.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    (root / f"{stem}.tex").write_text(latex.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    registry = _load(REGISTRY)
    entries = {entry["key"]: entry for entry in registry["entries"]}
    oracle = _load(ORACLE_STATS)
    operation_evidence = _load(ROOT / "outputs/reports/predicted_operation_evidence_v2.json")

    def value(key: str) -> Any:
        return entries[key]["expected_value"]

    t1_rows = []
    for operation in OPERATIONS:
        stat = oracle["operations"][operation]
        mean = stat["delta_iou"]["mean"]
        low = stat["delta_iou"]["ci95_low"]
        high = stat["delta_iou"]["ci95_high"]
        t1_rows.append(
            {
                "operation": operation,
                "label": LABELS[operation],
                "cases": int(value(f"oracle.{operation}.cases")),
                "delta_iou": mean,
                "mse": stat["delta_mse"]["mean"],
                "ci": (low, high),
                "outside": stat["outside_exact_ratio"]["mean"],
                "validity": float(value(f"oracle.{operation}.validity")),
            }
        )
    t1_md = [
        "# Main Oracle-Control Editing Anchor v1",
        "",
        "Status: frozen scoped release. CubiCasa5K test503 declared oracle controls;",
        "2,479 paired edits over five repeated execution seeds. Intervals resample",
        "floorplans, not duplicated seed rows.",
        "",
        "| Operation | N | Delta-IoU [95% CI] | Delta-MSE | Outside exact | Vector validity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in t1_rows:
        t1_md.append(
            f"| {row['label']} | {row['cases']} | {row['delta_iou']:.6f} "
            f"[{row['ci'][0]:.6f}, {row['ci'][1]:.6f}] | {row['mse']:.6f} | "
            f"{row['outside']:.6f} | {row['validity']:.6f} |"
        )
    t1_md.extend(
        [
            "",
            "Pooled FP-Eval validity is `1.0`; conditional watertightness is",
            "`0.957875-0.967491`. These are pooled five-operation diagnostics, not",
            "operation-specific estimates. FEVP five-check validity is `0.9124-0.9569`.",
            "Delta-IoU and Delta-MSE measure binary change support, not requested-class",
            "correctness or reference reconstruction fidelity.",
            "",
            f"<!-- registry bindings: {', '.join(registry['table_bindings']['T1_oracle_anchor'])} -->",
        ]
    )
    t1_tex_rows = "\n".join(
        f"{row['label']} & {row['cases']} & {row['delta_iou']:.6f} "
        f"[{row['ci'][0]:.6f}, {row['ci'][1]:.6f}] & {row['mse']:.6f} & "
        f"{row['outside']:.6f} & {row['validity']:.6f} \\\\"
        for row in t1_rows
    )
    t1_tex = rf"""\begin{{table*}}[t]
\centering
\caption{{Oracle-control editing anchor on CubiCasa5K test503. Intervals resample floorplans; repeated seeds are not independent plans. Delta metrics measure change support, not target-class correctness.}}
\label{{tab:oracle}}
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Operation & $N$ & $\Delta$IoU [95\% CI] & $\Delta$MSE & Outside exact & Vector validity \\
\midrule
{t1_tex_rows}
\bottomrule
\end{{tabular}}
\par\smallskip\footnotesize Pooled FP-Eval validity 1.000; conditional watertightness 0.9579--0.9675. FEVP five-check validity 0.9124--0.9569.
\end{{table*}}"""

    predicted_keys = [
        "raster_plausible",
        "plan_plausible",
        "room_coverage",
        "room_face_retention",
        "validity",
        "watertight",
        "functional_watertight",
        "wall_continuity_rate",
        "opening_wall_alignment",
        "furniture_room_containment",
        "outside_footprint_fraction",
        "room_overlap_ratio",
        "room_macro_iou_diagnostic",
    ]
    predicted_labels = {
        "raster_plausible": "Raster plausible",
        "plan_plausible": "Plan plausible",
        "room_coverage": "Room coverage",
        "room_face_retention": "Room-face retention",
        "validity": "FP-Eval validity",
        "watertight": "Raw watertight",
        "functional_watertight": "Functional watertight",
        "wall_continuity_rate": "Wall continuity",
        "opening_wall_alignment": "Opening-wall align",
        "furniture_room_containment": "Furniture containment",
        "outside_footprint_fraction": "Outside-footprint fraction",
        "room_overlap_ratio": "Room overlap ratio",
        "room_macro_iou_diagnostic": "Room macro IoU (diag.)",
    }
    predicted_values = {
        key: _mean(value(f"predicted.val491.{key}")) for key in predicted_keys
    }
    semantic = operation_evidence["semantic_diagnostics"]["operations"]
    semantic_rows = (
        ("Room add", "room_addition"),
        ("Room type", "room_type_change"),
        ("Room remove", "room_deletion"),
        ("Opening delete", "opening_pressure"),
        ("Furniture delete", "furniture_low_salience"),
    )
    t2_md = [
        "# Main Predicted-Control Reliability v1",
        "",
        "Status: frozen scoped release. Panel A uses CubiCasa val491 with five",
        "repeated target-free seeds. Panel B uses 738 development-exposed test503",
        "prompts. Semantic realization is a diagnostic from an older val191 model,",
        "not the selected release checkpoint.",
        "",
        "## Panel A: Structural reasonableness (val491, five seeds)",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    t2_md.extend(f"| {predicted_labels[key]} | {predicted_values[key]:.6f} |" for key in predicted_keys)
    t2_md.extend(
        [
            "",
            "Raw watertightness is `0.81670`, not `1.0`. Of 491 footprints, 401 are",
            "single-component and pass both raw and componentwise checks; 90 are",
            "multi-component and exceed the standard single-region FP-Eval boundary.",
            "Room macro IoU is a same-pipeline fidelity diagnostic and does not gate",
            "the reasonableness claim.",
            "",
            "## Panel B: Execution safety and semantic boundary",
            "",
            "| Evidence | Observed |",
            "|---|---:|",
            f"| Supported prompts executed | {operation_evidence['safety']['executed']} / {operation_evidence['safety']['prompts'] - operation_evidence['safety']['abstained']} |",
            f"| Unsupported/ambiguous abstained | {operation_evidence['safety']['abstained']} / {operation_evidence['safety']['abstained']} |",
            f"| Outside pixel exact (minimum) | {operation_evidence['safety']['outside_pixel_exact_min']:.6f} |",
            f"| Outside room-label exact (minimum) | {operation_evidence['safety']['outside_room_label_exact_min']:.6f} |",
            f"| Validity (minimum) | {operation_evidence['safety']['validity_min']:.6f} |",
            "",
            "| Semantic diagnostic | Successes / prompts | Rate |",
            "|---|---:|---:|",
        ]
    )
    for label, key in semantic_rows:
        row = semantic[key]
        t2_md.append(
            f"| {label} | {row['successful_prompts']} / {row['prompts']} | {row['success_rate']:.4f} |"
        )
    t2_md.extend(
        [
            "",
            "Object-delete and ghost-state audits pass their deterministic contracts;",
            "they are not semantic-success rows. No fresh held-out claim is made.",
            "",
            f"<!-- registry bindings: {', '.join(registry['table_bindings']['T2_predicted_reliability'])} -->",
        ]
    )
    predicted_tex_rows = "\n".join(
        f"{predicted_labels[key]} & {predicted_values[key]:.6f} \\\\" for key in predicted_keys
    )
    semantic_tex_rows = "\n".join(
        f"{label} & {semantic[key]['successful_prompts']}/{semantic[key]['prompts']} & "
        f"{semantic[key]['success_rate']:.4f} \\\\"
        for label, key in semantic_rows
    )
    t2_tex = rf"""\begin{{table*}}[t]
\centering
\caption{{Target-free predicted-control evidence. (A) Structural reasonableness on val491 across five repeated seeds. (B) Development-exposed safety prompts and a separately labeled older-model semantic diagnostic. Raw and functional watertightness are distinct.}}
\label{{tab:predicted}}
\small
\begin{{minipage}}[t]{{0.52\linewidth}}
\centering
\begin{{tabular}}{{lr}}
\toprule
Metric & Mean \\
\midrule
{predicted_tex_rows}
\bottomrule
\end{{tabular}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.45\linewidth}}
\centering
\begin{{tabular}}{{lrr}}
\toprule
Safety evidence & Observed & \\
\midrule
Supported executed & {operation_evidence['safety']['executed']}/{operation_evidence['safety']['prompts'] - operation_evidence['safety']['abstained']} & \\
Unsupported abstained & {operation_evidence['safety']['abstained']}/{operation_evidence['safety']['abstained']} & \\
Outside pixel exact & {operation_evidence['safety']['outside_pixel_exact_min']:.6f} & \\
Outside room exact & {operation_evidence['safety']['outside_room_label_exact_min']:.6f} & \\
Validity & {operation_evidence['safety']['validity_min']:.6f} & \\
\midrule
Semantic diagnostic & Success / $N$ & Rate \\
\midrule
{semantic_tex_rows}
\bottomrule
\end{{tabular}}
\end{{minipage}}
\par\smallskip\footnotesize Semantic rows use an older val191 model and are not a release gate. The safety population is development-exposed test, not held out.
\end{{table*}}"""

    t3_rows = []
    for operation in OPERATIONS:
        t3_rows.append(
            {
                "label": LABELS[operation],
                "cases": int(value(f"oracle.{operation}.cases")),
                "floorflow": _mean(value(f"oracle.{operation}.delta_iou")),
                "raster": float(value(f"oracle.{operation}.frozen_raster2seq_delta_iou_mean")),
                "editing": float(value(f"oracle.{operation}.matched_editing_aware_iou")),
                "gain_raster": float(value(f"oracle.{operation}.paired_vs_raster2seq_mean_delta")),
                "gain_editing": float(value(f"oracle.{operation}.paired_vs_editing_aware_mean_delta")),
            }
        )
    arms = (
        ("typed_edit_graph", "Typed routing"),
        ("dense_cross_attention", "Dense routing"),
        ("no_edit_gating", "No edit gating"),
        ("no_structural_graph", "No structural graph"),
    )
    t3_md = [
        "# Main Matched Controls and Composition Ablation v1",
        "",
        "Status: frozen scoped release. Same-case oracle-control comparisons; the",
        "Raster2Seq row is an outside-frozen redraw wrapper and the editing-aware",
        "control is intentionally weak. Neither is a native strong editing model.",
        "",
        "| Operation | N | FloorFlow | Raster2Seq wrapper | Editing-aware | Paired gain vs Raster2Seq | Paired gain vs editing-aware |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in t3_rows:
        t3_md.append(
            f"| {row['label']} | {row['cases']} | {row['floorflow']:.6f} | "
            f"{row['raster']:.6f} | {row['editing']:.6f} | {row['gain_raster']:.6f} | "
            f"{row['gain_editing']:.6f} |"
        )
    t3_md.extend(
        [
            "",
            "The frozen Raster2Seq wrapper has outside exactness `1.0`; this is a",
            "wrapper property, not native editing awareness. Paired confidence",
            "intervals use one difference per floorplan. The editing-aware runner",
            "recomputes its own paired FloorFlow row, so its absolute values can differ",
            "slightly from T1; comparisons stay within their own run.",
            "",
            "## Composition-policy ablation",
            "",
            "| Arm | Outside pixel | Outside room | Violation | Delta-IoU | Node F1 | Edge F1 | GED |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    arm_values: dict[str, dict[str, float]] = {}
    for arm, label in arms:
        key = "mechanism.typed_routing" if arm == "typed_edit_graph" else f"mechanism.{arm}"
        raw = value(key)
        row = {
            "outside_pixel": _mean(raw["outside_pixel_exact"]),
            "outside_room": _mean(raw["outside_room_exact"]),
            "violation": _mean(raw["locality_violation_rate"]),
            "delta_iou": _mean(raw["delta_iou"]),
            "node_f1": _mean(raw["node_f1"]),
            "edge_f1": _mean(raw["edge_f1"]),
            "ged": _mean(raw["ged"]),
        }
        arm_values[arm] = row
        t3_md.append(
            f"| {label} | {row['outside_pixel']:.6f} | {row['outside_room']:.6f} | "
            f"{row['violation']:.6f} | {row['delta_iou']:.6f} | {row['node_f1']:.6f} | "
            f"{row['edge_f1']:.6f} | {row['ged']:.6f} |"
        )
    typed_latency = float(value("mechanism.budget.typed_latency_ms"))
    no_graph_latency = float(value("mechanism.budget.no_graph_latency_ms"))
    iou_gain = float(value("mechanism.budget.delta_iou_gain"))
    ged_gain = float(value("mechanism.budget.ged_gain"))
    t3_md.extend(
        [
            "",
            "Arms are deterministic composition policies, not separately trained",
            "networks. At matched observable routing budget, typed routing adds",
            f"Delta-IoU `{iou_gain:+.6f}` and changes GED by `{ged_gain:+.6f}` with zero",
            "added learned parameters. It consumes more routing time",
            f"(`{typed_latency:.4f}` vs `{no_graph_latency:.4f}` ms mean); efficiency is",
            "not claimed. All 26 deterministic protocol attacks are sensitive.",
            "",
            f"<!-- registry bindings: {', '.join(registry['table_bindings']['T3_matched_and_ablation'])} -->",
        ]
    )
    t3_tex_rows = "\n".join(
        f"{row['label']} & {row['cases']} & {row['floorflow']:.6f} & {row['raster']:.6f} & "
        f"{row['editing']:.6f} & {row['gain_raster']:.6f} & {row['gain_editing']:.6f} \\\\"
        for row in t3_rows
    )
    t3_tex_arms = "\n".join(
        f"{label} & {arm_values[arm]['outside_pixel']:.6f} & {arm_values[arm]['outside_room']:.6f} & "
        f"{arm_values[arm]['violation']:.6f} & {arm_values[arm]['delta_iou']:.6f} & "
        f"{arm_values[arm]['node_f1']:.6f} & {arm_values[arm]['edge_f1']:.6f} & {arm_values[arm]['ged']:.6f} \\\\"
        for arm, label in arms
    )
    t3_tex = rf"""\begin{{table*}}[t]
\centering
\caption{{Same-case matched controls and deterministic composition-policy ablation. Raster2Seq is an outside-frozen wrapper and the editing-aware control is intentionally weak. The budget audit is post hoc and does not claim efficiency.}}
\label{{tab:matched}}
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Operation & $N$ & FloorFlow & Raster2Seq & Editing-aware & Gain vs R2S & Gain vs editing \\
\midrule
{t3_tex_rows}
\bottomrule
\end{{tabular}}
\par\medskip
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Arm & Outside px & Outside room & Violation & $\Delta$IoU & Node F1 & Edge F1 & GED \\
\midrule
{t3_tex_arms}
\bottomrule
\end{{tabular}}
\par\smallskip\footnotesize Typed routing: $\Delta$IoU gain {iou_gain:+.6f} and GED change {ged_gain:+.6f} at matched routing budget, with {typed_latency:.4f} vs {no_graph_latency:.4f} ms mean routing latency. Efficiency is not claimed.
\end{{table*}}"""

    house = {
        "delta_iou": float(value("external.housemind_e.delta_iou")),
        "delta_mse": float(value("external.housemind_e.delta_mse")),
        "node_f1": float(value("external.housemind_e.node_f1")),
        "edge": float(value("external.housemind_e.edge_overlap")),
        "ged": float(value("external.housemind_e.ged")),
    }
    fidelity = {
        key: float(value(f"predicted.fidelity.{key}"))
        for key in ("room_macro_iou", "node_f1", "edge_overlap", "ged")
    }
    typed_graph = arm_values["typed_edit_graph"]
    raster2seq_f1 = float(value("external.raster2seq.official_room_f1"))
    raster_values = [row["raster"] for row in t3_rows]
    oracle_iou = [row["delta_iou"] for row in t1_rows]
    oracle_mse = [row["mse"] for row in t1_rows]
    t4_md = [
        "# Main External Context v1",
        "",
        "Status: frozen context table. Rows use different datasets, inputs, and task",
        "protocols. No row forms a cross-dataset leaderboard.",
        "",
        "| Source | Task / input | Delta-IoU | Delta-MSE | Node F1 | Edge | GED | Allowed interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        f"| FloorFlow oracle control | CubiCasa test503; target-derived oracle controls | {min(oracle_iou):.6f}-{max(oracle_iou):.6f} | {min(oracle_mse):.6f}-{max(oracle_mse):.6f} | N/A | N/A | N/A | Scoped editing anchor; no leaderboard rank |",
        f"| FloorFlow typed diagnostic | 12,395 strict-locality rows | {typed_graph['delta_iou']:.6f} | N/A | {typed_graph['node_f1']:.6f} | {typed_graph['edge_f1']:.6f} | {typed_graph['ged']:.6f} | Composition-policy diagnostic |",
        f"| FloorFlow predicted fidelity | CubiCasa val491, target-free controls | N/A | N/A | {fidelity['node_f1']:.6f} | {fidelity['edge_overlap']:.6f} | {fidelity['ged']:.6f} | Reference fidelity remains weak |",
        f"| HouseMind-E | RPLAN natural-language editing, published | {house['delta_iou']:.6f} | {house['delta_mse']:.6f} | {house['node_f1']:.6f} | {house['edge']:.6f} | {house['ged']:.6f} | Published cross-dataset context only |",
        f"| Raster2Seq local redraw | CubiCasa edit rows; outside-frozen wrapper | {min(raster_values):.6f}-{max(raster_values):.6f} | N/A | N/A | N/A | N/A | Reconstruction wrapper, not native editing |",
        f"| Raster2Seq official | CubiCasa reconstruction, RoomF1 | N/A | N/A | N/A | N/A | N/A | RoomF1 `{raster2seq_f1:.6f}` public anchor |",
        "| HouseMind official code | Public assets | N/A | N/A | N/A | N/A | N/A | Serialization unavailable; no reproduction claim |",
        "",
        "FloorFlow oracle Delta-IoU and Delta-MSE numerically exceed the published",
        "HouseMind-E pair, but the datasets and input contracts differ. FloorFlow",
        "typed graph metrics remain below HouseMind-E graph metrics and predicted",
        "fidelity remains weak. No complete HouseMind defeat, RPLAN leaderboard rank,",
        "or full-CAD SOTA is claimed. Predicted room macro IoU remains 0.269701 and",
        "is reported as a limitation, not a hidden zero.",
        "",
        f"<!-- registry bindings: {', '.join(registry['table_bindings']['T4_external_context'])} -->",
    ]
    t4_tex = rf"""\begin{{table*}}[t]
\centering
\caption{{External context and fidelity boundary. Rows use different datasets, inputs, and protocols and must not be read as a leaderboard.}}
\label{{tab:external}}
\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{p{{0.13\linewidth}}p{{0.16\linewidth}}rrrrrp{{0.15\linewidth}}}}
\toprule
Source & Task / input & $\Delta$IoU & $\Delta$MSE & Node F1 & Edge & GED & Interpretation \\
\midrule
FloorFlow oracle & CubiCasa test503 oracle controls & {min(oracle_iou):.6f}--{max(oracle_iou):.6f} & {min(oracle_mse):.6f}--{max(oracle_mse):.6f} & -- & -- & -- & Scoped anchor \\
FloorFlow typed & 12,395 strict-locality rows & {typed_graph['delta_iou']:.6f} & -- & {typed_graph['node_f1']:.6f} & {typed_graph['edge_f1']:.6f} & {typed_graph['ged']:.6f} & Diagnostic \\
FloorFlow predicted & val491 target-free controls & -- & -- & {fidelity['node_f1']:.6f} & {fidelity['edge_overlap']:.6f} & {fidelity['ged']:.6f} & Weak fidelity \\
HouseMind-E & RPLAN editing, published & {house['delta_iou']:.6f} & {house['delta_mse']:.6f} & {house['node_f1']:.6f} & {house['edge']:.6f} & {house['ged']:.6f} & External context \\
Raster2Seq wrapper & CubiCasa frozen redraw & {min(raster_values):.6f}--{max(raster_values):.6f} & -- & -- & -- & -- & Reconstruction wrapper \\
\bottomrule
\end{{tabular}}
\par\smallskip\footnotesize FloorFlow oracle $\Delta$IoU/$\Delta$MSE numerically exceed the published HouseMind-E pair under different datasets and inputs. Graph and predicted-fidelity metrics are not uniformly superior; predicted room macro IoU is 0.269701.
\end{{table*}}"""

    _write("main_oracle_anchor_v1", "\n".join(t1_md), t1_tex)
    _write("main_predicted_reliability_v1", "\n".join(t2_md), t2_tex)
    _write("main_resource_and_ablation_v1", "\n".join(t3_md), t3_tex)
    _write("main_external_context_v1", "\n".join(t4_md), t4_tex)
    print(
        json.dumps(
            {
                "tables": [
                    "paper/tables/main_oracle_anchor_v1",
                    "paper/tables/main_predicted_reliability_v1",
                    "paper/tables/main_resource_and_ablation_v1",
                    "paper/tables/main_external_context_v1",
                ],
                "oracle_cases": sum(row["cases"] for row in t1_rows),
                "predicted_cases": 491,
                "semantic_prompts": sum(row["prompts"] for row in semantic.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
