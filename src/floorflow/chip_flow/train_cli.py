"""Command-line and model configuration for chip-flow training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.model_config import model_config
from floorflow.chip_flow.train_cli_options import add_architecture_training_args, add_structured_head_args
from floorflow.chip_flow.training_validation import validate_training_scope

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
    )
    parser.add_argument("--arm", default="slot_global")
    parser.add_argument(
        "--architecture",
        choices=(
            "global_slot",
            "graph_slot",
            "hierarchical_slot",
            "sparse_hierarchy",
            "slot_transport",
            "slot_residual",
            "trinity_token",
            "typed_hierarchy",
        ),
        default="global_slot",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="dev")
    parser.add_argument(
        "--train-indices",
        type=Path,
        default=None,
        help="Optional JSON list or {'indices': [...]} training subset.",
    )
    parser.add_argument(
        "--allow-full-data",
        action="store_true",
        help="Required when no frozen training subset is supplied.",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--sampling-mode",
        choices=("random", "balanced_stream", "complexity_tail_stream", "progressive_complexity"),
        default="random",
    )
    parser.add_argument(
        "--batch-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gpu-data-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep active training cache arrays resident on the training device.",
    )
    parser.add_argument(
        "--gpu-data-cache-max-gib",
        type=float,
        default=16.0,
        help="Maximum device memory allowed for the resident training cache.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-curve-seed", type=int, default=0)
    parser.add_argument("--eval-curve-manifest", type=Path, default=None)
    parser.add_argument("--data-manifest", type=Path, default=None)
    parser.add_argument("--response-sidecar", type=Path, default=None)
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=[0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000],
    )
    parser.add_argument("--curve-per-count", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--global-dim", type=int, default=192)
    parser.add_argument("--target-layers", type=int, default=4)
    parser.add_argument("--condition-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--trinity-depth", type=int, default=12)
    parser.add_argument("--hierarchy-depth", type=int, default=10)
    parser.add_argument("--typed-layers", type=int, default=1)
    parser.add_argument(
        "--relation-specific-attention",
        action="store_true",
        help="Use separate query-membership masks and gates per relation family.",
    )
    parser.add_argument(
        "--incidence-normalized-relations",
        action="store_true",
    )
    parser.add_argument(
        "--structural-relation-head",
        action="store_true",
        help="Predict grouping-aligned cluster geometry/connectivity relations.",
    )
    parser.add_argument("--group-moment-conditioning", action="store_true")
    parser.add_argument(
        "--complexity-conditioning",
        action="store_true",
        help="Inject explicit block/net/pin density features into the global token.",
    )
    parser.add_argument(
        "--structural-relation-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--relation-gate-weight", type=float, default=0.0)
    parser.add_argument(
        "--coarse-commit-fraction",
        type=float,
        default=None,
        help="Freeze L0/L1 after this fraction of the sampling trajectory.",
    )
    parser.add_argument("--coarse-anchor-tolerance", type=float, default=0.05)
    parser.add_argument("--coarse-anchor-strength", type=float, default=0.0)
    parser.add_argument("--coarse-anchor-max-step", type=float, default=0.1)
    parser.add_argument(
        "--typed-mode",
        choices=(
            "typed",
            "typed_bias",
            "dense",
            "no_structural",
            "no_constraint",
        ),
        default="typed",
    )
    parser.add_argument(
        "--use-hierarchy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-netlist",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--netlist-edge-gate", action="store_true")
    parser.add_argument("--dynamic-graph-relay", action="store_true")
    parser.add_argument(
        "--dynamic-graph-input",
        action="store_true",
        help="Condition hidden block tokens with current-geometry edge messages.",
    )
    parser.add_argument("--state-conditioned-edge-bias", action="store_true")
    parser.add_argument("--projector", action="store_true")
    parser.add_argument("--projector-iterations", type=int, default=4)
    parser.add_argument("--projector-boundary-axes", action="store_true")
    parser.add_argument(
        "--projector-sparse-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--projector-strategy",
        choices=("legacy", "conflict_aware"),
        default="legacy",
    )
    parser.add_argument("--projector-grid-size", type=int, default=16)
    parser.add_argument(
        "--projector-boundary-lock-corners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--graph-pe-k", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--min-slots", type=int, default=2)
    parser.add_argument("--assignment-top-k", type=int, default=2)
    parser.add_argument("--decoder-top-k", type=int, default=4)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--route-balance-weight", type=float, default=0.01)
    parser.add_argument("--routing-proxy-weight", type=float, default=0.0)
    parser.add_argument("--routing-proxy-final-weight", type=float, default=None)
    parser.add_argument("--routing-proxy-delay-steps", type=int, default=0)
    parser.add_argument("--routing-proxy-anneal-steps", type=int, default=0)
    parser.add_argument("--grouping-proxy-weight", type=float, default=0.0)
    parser.add_argument("--grouping-proxy-final-weight", type=float, default=None)
    parser.add_argument("--grouping-proxy-delay-steps", type=int, default=0)
    parser.add_argument("--grouping-proxy-anneal-steps", type=int, default=0)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument(
        "--output-mode",
        choices=("velocity", "x0"),
        default="velocity",
    )
    parser.add_argument(
        "--graph-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--graph-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--graph-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wire-dropout", type=float, default=0.0)
    parser.add_argument("--constraint-dropout", type=float, default=0.0)
    parser.add_argument("--global-pool-stats", action="store_true")
    parser.add_argument("--global-film", action="store_true")
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=1.0,
        help="Multiplier for the pretrained backbone learning rate.",
    )
    parser.add_argument(
        "--structured-head-lr-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--legalizer-head-lr-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--structured-head-warmup-steps", type=int, default=0)
    parser.add_argument("--legalizer-head-warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--grad-clip-per-group",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument(
        "--ema-mode",
        choices=("fixed", "warmup"),
        default="fixed",
    )
    parser.add_argument("--ema-warmup-steps", type=int, default=1000)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--official-cost-weight", type=float, default=0.0)
    parser.add_argument("--official-cost-violation-temperature", type=float, default=0.1)
    parser.add_argument("--official-cost-final-weight", type=float, default=None)
    parser.add_argument("--official-cost-delay-steps", type=int, default=0)
    parser.add_argument("--official-cost-anneal-steps", type=int, default=0)
    parser.add_argument("--physics-tail-weight", type=float, default=0.0)
    parser.add_argument("--physics-tail-fraction", type=float, default=0.25)
    parser.add_argument("--physics-tail-delay-steps", type=int, default=0)
    parser.add_argument("--physics-tail-anneal-steps", type=int, default=0)
    parser.add_argument("--overlap-weight-scale", type=float, default=1.0)
    parser.add_argument("--decode-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=0.1)
    parser.add_argument("--physics-final-weight", type=float, default=None)
    parser.add_argument("--decode-final-weight", type=float, default=None)
    parser.add_argument("--clean-final-weight", type=float, default=None)
    parser.add_argument("--weight-anneal-steps", type=int, default=0)
    parser.add_argument(
        "--weight-schedule",
        choices=("linear", "cosine"),
        default="linear",
    )
    add_architecture_training_args(parser)
    parser.add_argument("--path-steps", type=int, default=0)
    parser.add_argument("--path-weight", type=float, default=0.2)
    parser.add_argument("--nfe-consistency-weight", type=float, default=0.0)
    parser.add_argument("--nfe-consistency-steps", type=int, default=2)
    parser.add_argument("--nfe-consistency-target-weight", type=float, default=1.0)
    parser.add_argument("--nfe-consistency-self-weight", type=float, default=0.1)
    parser.add_argument("--rho-weight", type=float, default=0.0)
    parser.add_argument("--flow-huber-beta", type=float, default=0.1)
    parser.add_argument("--structured-head-contract", action="store_true")
    parser.add_argument(
        "--hierarchy-forced-head-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--endpoint-compatibility-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--training-projector", action="store_true")
    parser.add_argument("--training-projector-iterations", type=int, default=2)
    parser.add_argument("--training-projector-strength", type=float, default=1.0)
    parser.add_argument(
        "--training-projector-displacement-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--endpoint-compatibility-iterations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--endpoint-compatibility-displacement-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--endpoint-compatibility-gain-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--endpoint-refiner-energy-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--endpoint-refiner-trajectory-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--endpoint-refiner-trajectory-lr",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument("--endpoint-refiner-trajectory-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-legalized-area-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-legalized-p2b-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-legalizer-target-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-mib-share-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-pin-tail-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-rho-trust-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-rho-trust-radius", type=float, default=0.1)
    parser.add_argument("--endpoint-pin-tail-fraction", type=float, default=0.2)
    parser.add_argument("--legalizer-response-weight", type=float, default=0.0)
    parser.add_argument("--legalizer-displacement-weight", type=float, default=1.0)
    parser.add_argument("--legalizer-rung-weight", type=float, default=0.1)
    parser.add_argument("--legalizer-response-delay-steps", type=int, default=0)
    parser.add_argument("--legalizer-response-anneal-steps", type=int, default=0)
    parser.add_argument(
        "--endpoint-compatibility-delay-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--endpoint-compatibility-anneal-steps",
        type=int,
        default=0,
    )
    parser.add_argument("--hierarchy-global-weight", type=float, default=0.1)
    parser.add_argument("--hierarchy-structural-weight", type=float, default=0.1)
    parser.add_argument("--hierarchy-port-weight", type=float, default=0.1)
    parser.add_argument("--hierarchy-cycle-weight", type=float, default=0.0)
    parser.add_argument("--projector-aware-endpoint", action="store_true")
    parser.add_argument(
        "--projector-aware-endpoint-iterations",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--projector-aware-endpoint-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument("--projector-residual-head", action="store_true")
    parser.add_argument(
        "--projector-residual-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--projector-residual-iterations",
        type=int,
        default=4,
    )
    parser.add_argument("--hierarchy-bbox-weight", type=float, default=0.0)
    parser.add_argument(
        "--hierarchy-bbox-decay-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hierarchy-outline-conditioning",
        action="store_true",
    )
    parser.add_argument(
        "--hierarchy-order-conditioning",
        action="store_true",
    )
    parser.add_argument(
        "--hierarchy-coarse-to-fine",
        action="store_true",
    )
    parser.add_argument(
        "--hierarchy-port-consensus",
        action="store_true",
    )
    parser.add_argument(
        "--hierarchy-port-output",
        action="store_true",
        help="Add port consensus directly to the final x0 prediction.",
    )
    parser.add_argument(
        "--hierarchy-boundary-anchor",
        action="store_true",
        help="Correct final x0 geometry for boundary-coded blocks only.",
    )
    parser.add_argument(
        "--trinity-backbone-init",
        type=Path,
        default=None,
        help="Strictly initialize backbone weights from a pinned Trinity release.",
    )
    parser.add_argument(
        "--hierarchy-spatial-occupancy",
        action="store_true",
    )
    parser.add_argument(
        "--spatial-route-field",
        action="store_true",
        help="Add block extents, net demand and pin-demand channels.",
    )
    parser.add_argument("--symmetric-b2b-messages", action="store_true")
    parser.add_argument(
        "--hierarchy-complexity-slots",
        action="store_true",
    )
    parser.add_argument(
        "--degree-conditioned-expert",
        action="store_true",
    )
    parser.add_argument("--global-spec-conditioning", action="store_true")
    parser.add_argument("--node-semantic-dim", type=int, default=0)
    parser.add_argument("--port-semantic-dim", type=int, default=0)
    parser.add_argument("--netlist-hyperedge-conditioning", action="store_true")
    parser.add_argument(
        "--membership-hierarchy-conditioning",
        action="store_true",
    )
    parser.add_argument(
        "--standalone-legalizer-response",
        action="store_true",
    )
    parser.add_argument("--hierarchy-coordinate-parameterization", action="store_true")
    parser.add_argument(
        "--hierarchy-coordinate-mode",
        choices=("envelope", "envelope_cluster"),
        default="envelope_cluster",
    )
    parser.add_argument("--observed-hierarchy-coupling", action="store_true")
    parser.add_argument(
        "--hierarchy-token-bridge",
        action="store_true",
    )
    parser.add_argument(
        "--hierarchy-token-bridge-mode",
        choices=("legacy", "bounded", "structural"),
        default="legacy",
    )
    parser.add_argument("--global-bbox-consensus", action="store_true")
    parser.add_argument("--self-conditioning", action="store_true")
    add_structured_head_args(parser)
    parser.add_argument("--backbone-lora-rank", type=int, default=0)
    parser.add_argument("--backbone-lora-alpha", type=float)
    parser.add_argument(
        "--self-conditioning-keep-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hierarchy-outline-max-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--hierarchy-source-mode",
        choices=("random", "shelf", "noisy_shelf", "mixed"),
        default="random",
        help=(
            "Block-state source during training. shelf matches deployment "
            "sampling; random preserves legacy checkpoints."
        ),
    )
    parser.add_argument(
        "--hierarchy-source-noise",
        type=float,
        default=0.0,
        help="Gaussian noise added to shelf sources in noisy_shelf mode.",
    )
    parser.add_argument(
        "--hierarchy-source-shelf-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hierarchy-source-shelf-probability-final",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hierarchy-source-shelf-probability-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hierarchy-source-max-shelf-probability",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--hierarchy-source-levels",
        choices=("random", "derived"),
        default="random",
        help=(
            "Initialize L0/L1/L3 hierarchy source states randomly or derive "
            "them consistently from the shelf block source."
        ),
    )
    parser.add_argument(
        "--require-deployment-source",
        action="store_true",
        help="Reject random block or hierarchy source policies.",
    )
    parser.add_argument("--time-std", type=float, default=1.0)
    parser.add_argument(
        "--geometry-augment",
        choices=("none", "hflip", "vflip", "rot180", "dihedral"),
        default="none",
    )
    parser.add_argument(
        "--geometry-augment-probability",
        type=float,
        default=1.0,
    )
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--full-dev-eval", action="store_true")
    parser.add_argument(
        "--eval-state",
        choices=("raw", "ema"),
        default="ema",
        help="Model state used for dev checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Optional data-order seed; defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-weights-only", action="store_true")
    parser.add_argument("--resume-allow-new-params", action="store_true")
    parser.add_argument("--freeze-existing", action="store_true")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the initialized backbone while training typed adapters.",
    )
    parser.add_argument(
        "--backbone-unfreeze-step",
        type=int,
        default=None,
        help="Freeze the backbone, then unfreeze it at this optimizer step.",
    )
    parser.add_argument("--overlap-threshold", type=float, default=0.05)
    parser.add_argument("--total-threshold", type=float, default=0.030)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("final", "best", "legalized_best"),
        default="final",
    )
    parser.add_argument("--checkpoint-shortlist-size", type=int, default=0)
    parser.add_argument("--checkpoint-selection-report", type=Path, default=None)
    return parser.parse_args()


def json_args(args: argparse.Namespace) -> dict:
    return json.loads(json.dumps(vars(args), default=str))


__all__ = [
    "REPO_ROOT",
    "json_args",
    "model_config",
    "parse_args",
    "validate_training_scope",
]
