from types import SimpleNamespace

import pytest
import torch
from torch import nn

from floorflow.representation.tensor_checks import assert_true_async
from floorflow.training.gpu_profiler import profile_training_epoch
from floorflow.training.metric_accumulator import (
    MetricAccumulator,
    PackedMetricAccumulator,
)
from floorflow.training.runtime_acceleration import RuntimeConfig, configure_torch_runtime


def test_cpu_runtime_falls_back_to_fp32_context() -> None:
    cpu_device = torch.device("cpu")
    runtime = RuntimeConfig(precision="bf16")
    with runtime.autocast(cpu_device):
        value = torch.ones(2) + 1
    assert value.dtype == torch.float32
    assert not runtime.scaler_enabled


def test_dataloader_options_avoid_invalid_worker_settings() -> None:
    cpu_device = torch.device("cpu")
    runtime = RuntimeConfig(prefetch_factor=3)
    options = runtime.dataloader_options(cpu_device, workers=0)
    assert options == {"num_workers": 0, "pin_memory": False}


def test_runtime_rejects_invalid_precision_and_prefetch() -> None:
    with pytest.raises(ValueError, match="precision"):
        RuntimeConfig(precision="tf32")
    with pytest.raises(ValueError, match="prefetch"):
        RuntimeConfig(prefetch_factor=0)


def test_runtime_configuration_is_serializable() -> None:
    cpu_device = torch.device("cpu")
    configure_torch_runtime(cpu_device)
    manifest = RuntimeConfig(precision="fp16").manifest()
    assert manifest["precision"] == "fp16"
    assert manifest["compile_model"] is False
    assert manifest["triton_kernels"] is False


def test_runtime_config_honors_compile_options() -> None:
    args = SimpleNamespace(
        precision="bf16",
        compile_model=True,
        compile_mode="reduce-overhead",
        channels_last=True,
        prefetch_factor=2,
    )
    runtime = RuntimeConfig.from_args(args)
    assert runtime.compile_model is True
    assert runtime.compile_mode == "reduce-overhead"
    assert runtime.manifest()["channels_last"] is True
    functorch_config = getattr(torch._functorch, "config", None)
    if functorch_config is None or not hasattr(
        functorch_config,
        "donated_buffer",
    ):
        pytest.skip("this Torch build does not expose Functorch config")
    original = functorch_config.donated_buffer
    try:
        model = runtime.prepare_model(nn.Linear(2, 2))
        assert isinstance(model, nn.Module)
        assert functorch_config.donated_buffer is False
        with runtime.eager_model(model) as eager_model:
            assert eager_model is model
            assert model._compiled_call_impl is None
            assert model(torch.ones(1, 2)).shape == (1, 2)
        assert model._compiled_call_impl is not None
    finally:
        functorch_config.donated_buffer = original


def test_async_check_and_profiler_have_safe_cpu_contracts(tmp_path) -> None:
    assert_true_async(torch.tensor(True), "must hold")
    with pytest.raises(ValueError, match="must hold"):
        assert_true_async(torch.tensor(False), "must hold")
    with profile_training_epoch(torch.device("cpu"), tmp_path) as profiler:
        value = torch.ones(4) @ torch.ones(4)
        assert value == 4
        for _ in range(4):
            profiler.step()
    assert (tmp_path / "trace.json").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_metric_accumulator_finalizes_tensor_and_float_totals() -> None:
    totals = MetricAccumulator()
    totals.add("loss", torch.tensor(2.0), weight=2)
    totals.add("loss", torch.tensor(4.0), weight=3)
    totals.add("count", torch.tensor([1.0, 3.0]), weight=2)
    totals.add("scalar", 5.0, weight=2)
    result = totals.finalize(5)
    assert result["loss"] == pytest.approx(3.2)
    assert result["count"] == pytest.approx(0.8)
    assert result["scalar"] == pytest.approx(2.0)


def test_packed_metric_accumulator_matches_scalar_contract() -> None:
    totals = PackedMetricAccumulator()
    totals.add({"loss": torch.tensor(2.0), "count": 1.0}, weight=2)
    totals.add({"loss": torch.tensor(4.0), "late": torch.tensor(3.0)}, weight=3)
    result = totals.finalize(5)
    assert result["loss"] == pytest.approx(3.2)
    assert result["count"] == pytest.approx(0.4)
    assert result["late"] == pytest.approx(1.8)
