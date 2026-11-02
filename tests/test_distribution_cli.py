from types import SimpleNamespace

from floorflow.training.distribution_cli import (
    validate_distribution_runtime_args,
)


def test_cuda_graph_compile_and_gradient_balancing_are_rejected() -> None:
    args = SimpleNamespace(
        compile_model=True,
        compile_mode="reduce-overhead",
        gradient_balance_interval=50,
    )
    assert "incompatible" in validate_distribution_runtime_args(args)


def test_compile_without_gradient_balancing_is_allowed() -> None:
    args = SimpleNamespace(
        compile_model=True,
        compile_mode="default",
        gradient_balance_interval=0,
    )
    assert validate_distribution_runtime_args(args) is None


def test_default_compile_is_rejected_with_gradient_balancing() -> None:
    args = SimpleNamespace(
        compile_model=True,
        compile_mode="default",
        gradient_balance_interval=50,
    )
    assert "incompatible" in validate_distribution_runtime_args(args)
