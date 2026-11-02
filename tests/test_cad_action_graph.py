import torch
import torch.nn.functional as F

from floorflow.model.cad_action_graph import (
    PREDECESSORS,
    STAGES,
    CadActionGraph,
)


def test_cad_action_graph_is_ordered_and_typed() -> None:
    graph = CadActionGraph(
        {name: 4 for name in STAGES},
        hidden_dim=16,
        heads=4,
    )
    inputs = {
        name: torch.randn(2, 4)
        for name in STAGES
    }
    with torch.no_grad():
        graph.update_scale.fill_(0.1)
    output = graph(inputs)
    assert tuple(output.tokens) == STAGES
    assert output.link_prior.shape == (len(STAGES), len(STAGES))
    assert output.link_prior[2, 0] > output.link_prior[2, 7]
    assert output.link_prior[5, 4] > output.link_prior[3, 5]
    for value in output.gates.values():
        assert value.shape == (2,)
        assert torch.all(value >= 0.0)
        assert torch.all(value <= 1.0)
    loss = sum(value.square().mean() for value in output.tokens.values())
    loss.backward()
    assert graph.link_residual.grad is not None
    assert torch.isfinite(graph.link_residual.grad).all()


def test_predecessors_follow_cad_action_order() -> None:
    assert PREDECESSORS["rooms"] == ("envelope", "program")
    assert PREDECESSORS["walls"] == ("envelope", "program", "rooms")
    assert PREDECESSORS["openings"] == ("rooms", "walls", "junctions")
    assert PREDECESSORS["furniture"] == ("program", "rooms")


def test_spatial_action_token_retains_salient_peaks() -> None:
    graph = CadActionGraph(
        {name: 1 for name in STAGES},
        hidden_dim=16,
        heads=4,
    )
    field = torch.zeros(1, 1, 8, 8)
    field[:, :, 3:5, 3:5] = 4.0

    token = graph._token("walls", field)
    pooled = 0.5 * (
        F.adaptive_avg_pool2d(field, (1, 1))
        + F.adaptive_max_pool2d(field, (1, 1))
    ).flatten(1)
    expected = graph.projections["walls"](pooled)

    torch.testing.assert_close(token, expected)


def test_cad_action_graph_starts_with_small_causal_residuals() -> None:
    graph = CadActionGraph(
        {name: 4 for name in STAGES},
        hidden_dim=16,
        heads=4,
        route_attention=True,
        route_router_count=4,
        route_top_k=2,
    )
    inputs = {name: torch.randn(2, 4) for name in STAGES}

    output = graph(inputs)

    torch.testing.assert_close(
        output.tokens["envelope"],
        graph.projections["envelope"](inputs["envelope"]),
    )
    for name in STAGES[1:]:
        assert not torch.allclose(
            output.tokens[name],
            graph.projections[name](inputs[name]),
        )


def test_cad_action_graph_can_add_causal_mita_global_context() -> None:
    graph = CadActionGraph(
        {name: 4 for name in STAGES},
        hidden_dim=16,
        heads=4,
        route_attention=True,
        route_router_count=4,
        route_top_k=2,
    )
    inputs = {name: torch.randn(2, 4) for name in STAGES}
    with torch.no_grad():
        graph.update_scale.fill_(0.1)
        graph.route_scale.fill_(0.1)
    output = graph(inputs)
    assert output.route_diagnostics is not None
    assert set(output.route_diagnostics) == set(STAGES) - {"envelope"}
    diagnostics = output.route_diagnostics["furniture"]
    assert torch.isfinite(diagnostics["routing_entropy"])
    assert torch.isfinite(diagnostics["route_budget_soft_mean"])
    sum(value.square().mean() for value in output.tokens.values()).backward()
    assert graph.route is not None
    assert graph.route.q_proj.weight.grad is not None
    assert graph.route_budget["furniture"][0].weight.grad is not None
