from __future__ import annotations

import torch

from floorflow.representation.program_conditioning import (
    ProgramConditioner,
    apply_program_constraint,
    assign_instances_to_program,
)


def test_program_constraint_suppresses_absent_classes() -> None:
    logits = torch.zeros(2, 4, 8, 8)
    program = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    constrained = apply_program_constraint(logits, program, absent_penalty=10.0)
    assert (constrained[:, [1, 3]].min() <= -9.9)
    assert (constrained[0, [0, 2]].abs() < 1.0e-6).all()
    assert (constrained[1, [1, 3]].abs() < 1.0e-6).all()


def test_program_conditioner_bias_is_differentiable() -> None:
    module = ProgramConditioner(5, hidden_dim=16)
    program = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]], requires_grad=True)
    output = module(program)
    assert output["class_bias"].shape == (1, 5)
    output["class_bias"].sum().backward()
    assert program.grad is not None


def test_instance_program_assignment_respects_counts() -> None:
    labels = torch.zeros(1, 8, 16, dtype=torch.long)
    labels[:, :, :7] = 10
    labels[:, :, 9:] = 10
    logits = torch.zeros(1, 2, 8, 16)
    counts = torch.tensor([[1, 1]])
    output = assign_instances_to_program(labels, logits, counts, (10, 20), min_area=1)
    left = set(output[0, :, :7].flatten().tolist())
    right = set(output[0, :, 9:].flatten().tolist())
    assert left | right == {10, 20}
