#!/usr/bin/env bash
set -euo pipefail

cuda_device="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${cuda_device}"

uv run python scripts/evaluate_distribution_overlay.py \
  --checkpoint outputs/train/distribution_retrieval_free_merged_style_v2.pt \
  --room-segmenter-checkpoint outputs/train/room_segmenter_crop_consistency_v1/best.pt \
  --recursion-policy-checkpoint outputs/train/recursion_policy_v1/best.pt \
  --recursive-graph-checkpoint outputs/train/recursive_graph_adapter_1024_v1/best.pt \
  --room-graph-trust-checkpoint outputs/train/room_graph_trust_gate_smoke_v1/best.pt \
  --room-gate-decouple \
  --room-graph-confidence-threshold 0.2 \
  --room-graph-trust-threshold 0.35 \
  --furniture-query-checkpoint outputs/train/dynamic_furniture_queries_1024_v2/best.pt \
  --furniture-query-threshold 0.7 \
  --recursive-context-fraction 0.75 \
  --recursive-blend 0.1 \
  --room-prior-weight 3.0 \
  --room-prior-confidence-threshold 1.0 \
  --retrieval-prior-weight 0 \
  --style-id 0 \
  --split test \
  --offset 0 \
  --limit 503 \
  --batch-size 4 \
  --workers 4 \
  --device cuda \
  --overlay-wall-threshold 0.3 \
  --overlay-opening-threshold 0.2 \
  --overlay-furniture-threshold 0.2 \
  --output outputs/reports/final_retrieval_free_style_test503_v1.json \
  --per-case-output outputs/reports/final_retrieval_free_style_per_case_test503_v1.json

uv run python scripts/analyze_recursive_decisions.py \
  --per-case outputs/reports/final_retrieval_free_style_per_case_test503_v1.json \
  --strata outputs/data/cubicas_strata_v2.jsonl \
  --split test \
  --offset 0 \
  --limit 503 \
  --output outputs/reports/final_retrieval_free_style_decision_audit_test503_v1.json
