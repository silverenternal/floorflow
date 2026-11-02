"""Dynamic kernels and query heads adapted from local EoMT/MiTA sources."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.model.official_mita_route import OfficialMiTARouteAttention


def _odd_pairs(
    kernels: Iterable[int] | Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    pairs = []
    for value in kernels:
        kernel, dilation = value if isinstance(value, tuple) else (value, 1)
        kernel, dilation = int(kernel), int(dilation)
        if kernel < 1 or kernel % 2 == 0 or dilation < 1:
            raise ValueError("kernels must be positive odd sizes and dilations positive")
        pairs.append((kernel, dilation))
    if not pairs:
        raise ValueError("at least one kernel configuration is required")
    return tuple(sorted(set(pairs)))


class GeneratedDepthwiseKernel(nn.Module):
    """Generate a sample-specific depthwise kernel from pooled context."""

    def __init__(self, channels: int, kernel_size: int, bases: int = 4) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.bases = int(bases)
        router_channels = max(16, self.channels // 4)
        self.basis = nn.Parameter(
            torch.randn(self.bases, 1, kernel_size, kernel_size)
            * (1.0 / math.sqrt(kernel_size * kernel_size))
        )
        self.generator = nn.Sequential(
            nn.Linear(self.channels * 2, router_channels),
            nn.SiLU(),
            nn.Linear(router_channels, self.bases),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.cat(
            [features.mean(dim=(2, 3)), features.amax(dim=(2, 3))],
            dim=1,
        )
        coefficients = self.generator(context).softmax(dim=1)
        kernel = torch.einsum("bc,ckl->bkl", coefficients, self.basis[:, 0])
        return kernel, coefficients

    def apply(self, features: torch.Tensor, dilation: int = 1) -> torch.Tensor:
        generated, _ = self(features)
        batch, channels, height, width = features.shape
        weight = generated[:, None].expand(-1, channels, -1, -1)
        weight = weight.reshape(batch * channels, 1, *generated.shape[-2:])
        flattened = features.reshape(1, batch * channels, height, width)
        padding = dilation * (self.kernel_size - 1) // 2
        output = F.conv2d(
            flattened,
            weight,
            padding=padding,
            dilation=dilation,
            groups=batch * channels,
        )
        return output.reshape(batch, channels, height, width)


class DynamicKernelOutputHead(nn.Module):
    """Sample-conditioned kernels with learned per-pixel scale routing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_configs: Iterable[int | tuple[int, int]] = (
            (1, 1),
            (3, 1),
            (5, 1),
            (7, 1),
            (3, 2),
            (5, 2),
        ),
        kernel_bases: int = 4,
    ) -> None:
        super().__init__()
        self.configs = _odd_pairs(kernel_configs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.generators = nn.ModuleList(
            [
                GeneratedDepthwiseKernel(in_channels, kernel, kernel_bases)
                for kernel, _dilation in self.configs
            ]
        )
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1) for _ in self.configs]
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        router_channels = max(16, in_channels // 2)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, router_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                router_channels,
                out_channels * len(self.configs),
                1,
            ),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.last_scale_weights: torch.Tensor | None = None
        self.last_scale_weights_raw: torch.Tensor | None = None
        self.last_kernel_coefficients: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch_outputs = []
        coefficients = []
        for generator, projection, (_kernel, dilation) in zip(
            self.generators,
            self.projections,
            self.configs,
        ):
            branch_outputs.append(projection(generator.apply(features, dilation)))
            coefficients.append(generator(features)[1])
        branches = torch.stack(branch_outputs, dim=1)
        batch, candidates, channels, height, width = branches.shape
        logits = self.router(features).reshape(
            batch,
            candidates,
            channels,
            height,
            width,
        )
        weights = logits.softmax(dim=1)
        self.last_scale_weights = weights.detach()
        self.last_scale_weights_raw = weights
        self.last_kernel_coefficients = torch.stack(coefficients, dim=1).detach()
        return (weights * branches).sum(dim=1)


class DynamicQueryMaskHead(nn.Module):
    """EoMT-style learned query masks with a learned query budget."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_queries: int = 16,
        hidden_channels: int = 96,
        token_grid: int = 8,
    ) -> None:
        super().__init__()
        self.max_queries = int(max_queries)
        self.token_grid = int(token_grid)
        self.memory = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(
            torch.randn(self.max_queries, hidden_channels) * 0.02
        )
        self.route = OfficialMiTARouteAttention(
            hidden_channels,
            num_heads=4,
            router_count=min(16, self.max_queries),
            top_k=min(8, self.max_queries),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.memory_projection = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            1,
        )
        self.class_head = nn.Linear(hidden_channels, out_channels + 1)
        self.budget_head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.budget_head.weight)
        nn.init.zeros_(self.budget_head.bias)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(
        self,
        features: torch.Tensor,
        minimum_queries: int = 1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory = F.adaptive_avg_pool2d(
            self.memory(features),
            (self.token_grid, self.token_grid),
        )
        tokens = memory.flatten(2).transpose(1, 2)
        queries = self.queries[None].expand(features.shape[0], -1, -1)
        query, route_diag = self.route(queries, tokens)
        class_logits = self.class_head(query)
        budget_probability = self.budget_head(query.mean(dim=1)).sigmoid()[:, 0]
        budget = (
            minimum_queries
            + budget_probability * (self.max_queries - minimum_queries)
        )
        gate = torch.sigmoid(
            self.budget_head(query)[..., 0]
            + torch.linspace(
                1.0,
                -1.0,
                self.max_queries,
                device=features.device,
            )
        )
        budget_gate = gate * (torch.arange(self.max_queries, device=features.device)[None] < budget[:, None])
        masks = torch.einsum(
            "bqc,bchw->bqhw",
            self.mask_head(query),
            self.memory_projection(memory),
        )
        weights = class_logits[..., :-1].softmax(dim=1)
        weighted_masks = torch.einsum("bqhw,bqc->bqchw", masks, weights)
        weighted_masks = weighted_masks * budget_gate[:, :, None, None, None]
        denominator = budget_gate.sum(dim=1)[:, None, None, None]
        logits = weighted_masks.sum(dim=1) / denominator.clamp_min(1.0)
        logits = F.interpolate(
            logits,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        diagnostics = {
            **route_diag,
            "query_budget": budget.detach(),
            "query_gate_mean": budget_gate.detach().mean(),
            "query_usage": budget_gate.detach().mean(dim=0),
        }
        self.last_diagnostics = diagnostics
        return logits, diagnostics


def dynamic_routing_loss(
    weights: torch.Tensor,
    *,
    target_entropy: float = 0.7,
    load_balance_weight: float = 0.01,
    entropy_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regularize candidate usage and confidence without a fixed route."""

    if weights.ndim < 2:
        raise ValueError("routing weights need a candidate dimension")
    probability = weights.float()
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        1.0e-6
    )
    usage = probability.mean(dim=tuple(range(2, probability.ndim)))
    dead_expert_fraction = (usage.detach() < 1.0e-4).float().mean()
    balance = usage.square().sum(dim=1).mean()
    entropy = -(
        probability.clamp_min(1.0e-8) * probability.clamp_min(1.0e-8).log()
    ).sum(dim=1)
    entropy = entropy / math.log(probability.shape[1])
    entropy_penalty = (entropy.mean() - float(target_entropy)).square()
    total = (
        float(load_balance_weight) * (balance - 1.0 / probability.shape[1])
        + float(entropy_weight) * entropy_penalty
    )
    return total, {
        "routing_balance": balance.detach(),
        "routing_entropy": entropy.mean().detach(),
        "routing_entropy_penalty": entropy_penalty.detach(),
        "dead_expert_fraction": dead_expert_fraction,
    }


__all__ = [
    "DynamicKernelOutputHead",
    "DynamicQueryMaskHead",
    "GeneratedDepthwiseKernel",
    "dynamic_routing_loss",
]
