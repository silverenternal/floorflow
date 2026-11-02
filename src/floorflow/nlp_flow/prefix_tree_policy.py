"""Prompt-conditioned causal-prefix transformer for typed AST actions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from floorflow.nlp_flow.engineering.schedule_free import ScheduleFreeAdamW
from floorflow.nlp_flow.prefix_policy_data import PrefixPolicyBatch
from floorflow.nlp_flow.real_corpus_policy_data import (
    MAX_LEXEME,
    MAX_NODES,
    MAX_PROMPT,
    MAX_PROMPT_TOKENS,
    PROMPT_TOKEN_BUCKETS,
    REAL_ACTIONS,
    STATE_FEATURES,
)


class PrefixTreePolicy(nn.Module):
    def __init__(
        self,
        vocabulary: Mapping[str, object],
        *,
        hidden_size: int = 128,
        layers: int = 4,
        heads: int = 8,
        max_steps: int = 768,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.vocabulary = dict(vocabulary)
        self.hidden_size = hidden_size
        self.prompt_token_embedding = nn.Embedding(
            PROMPT_TOKEN_BUCKETS + 1,
            hidden_size,
            padding_idx=0,
        )
        self.prompt_position_embedding = nn.Embedding(
            MAX_PROMPT_TOKENS,
            hidden_size,
        )
        prompt_layer = nn.TransformerEncoderLayer(
            hidden_size,
            heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(
            prompt_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.prompt_norm = nn.RMSNorm(hidden_size)

        self.previous_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(vocabulary[key]) + 1, hidden_size)
                for key in (
                    "action_ids",
                    "node_ids",
                    "edge_ids",
                    "class_ids",
                    "lexeme_ids",
                    "gap_ids",
                )
            ]
        )
        self.history_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(vocabulary[key]) + 1, hidden_size)
                for key in (
                    "action_ids",
                    "node_ids",
                    "edge_ids",
                    "class_ids",
                    "lexeme_ids",
                    "gap_ids",
                )
            ]
        )
        self.state_projection = nn.Linear(STATE_FEATURES, hidden_size)
        self.context_kind_embedding = nn.Embedding(
            len(vocabulary["node_ids"]) + 1,
            hidden_size,
        )
        self.context_edge_embedding = nn.Embedding(
            len(vocabulary["edge_ids"]) + 1,
            hidden_size,
        )
        self.start_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.step_position_embedding = nn.Embedding(max_steps, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size,
            heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=layers,
        )
        self.decoder_norm = nn.RMSNorm(hidden_size)

        self.prompt_char_embedding = nn.Embedding(
            len(vocabulary["chars"]) + 1,
            24,
        )
        self.prompt_char_position = nn.Embedding(MAX_PROMPT, hidden_size)
        self.prompt_char_projection = nn.Linear(24, hidden_size)
        self.copy_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.parent_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.candidate_node_embedding = nn.Embedding(
            len(vocabulary["node_ids"]) + 1,
            hidden_size,
        )
        self.candidate_scope_embedding = nn.Embedding(
            len(vocabulary["scope_ids"]) + 1,
            hidden_size,
        )
        self.candidate_position_embedding = nn.Embedding(MAX_NODES, hidden_size)
        self.candidate_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(hidden_size, len(REAL_ACTIONS)),
                "node_kind": nn.Linear(hidden_size, len(vocabulary["node_ids"]) + 1),
                "edge_field": nn.Linear(hidden_size, len(vocabulary["edge_ids"]) + 1),
                "leaf_node": nn.Linear(hidden_size, MAX_NODES),
                "leaf_class": nn.Linear(hidden_size, len(vocabulary["class_ids"]) + 1),
                "copy": nn.Linear(hidden_size, 1),
                "copy_length": nn.Linear(hidden_size, MAX_LEXEME),
                "lexical": nn.Linear(hidden_size, len(vocabulary["lexeme_ids"]) + 1),
                "gap": nn.Linear(hidden_size, len(vocabulary["gap_ids"]) + 1),
                "node_budget": nn.Linear(hidden_size, MAX_NODES + 1),
            }
        )

    def encode_prompt(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        chars: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        prompt = (
            self.prompt_token_embedding(token_ids)
            + self.prompt_position_embedding(positions)[None]
        )
        memory = self.prompt_norm(
            self.prompt_encoder(
                prompt,
                src_key_padding_mask=~token_mask,
            )
        )
        weights = token_mask.to(memory.dtype)
        pooled = (memory * weights[..., None]).sum(dim=1) / weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)
        char_positions = torch.arange(chars.shape[1], device=chars.device)
        char_states = (
            self.prompt_char_projection(self.prompt_char_embedding(chars))
            + self.prompt_char_position(char_positions)[None]
        )
        return memory, pooled, char_states

    def forward(
        self,
        batch: PrefixPolicyBatch,
        *,
        steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        sequence_steps = batch.action.shape[1] if steps is None else steps
        sequence_steps = min(sequence_steps, batch.action.shape[1])
        memory, pooled, char_states = self.encode_prompt(
            batch.prompt_token_ids,
            batch.prompt_token_mask,
            batch.prompts,
            batch.prompt_masks,
        )
        previous = sum(
            embedding(batch.previous[key][:, :sequence_steps])
            for embedding, key in zip(
                self.previous_embeddings,
                ("action", "node", "edge", "leaf_class", "lexical", "gap"),
                strict=True,
            )
        )
        action_history = sum(
            embedding(batch.history[key][:, :sequence_steps]).mean(dim=-2)
            for embedding, key in zip(
                self.history_embeddings,
                ("action", "node", "edge", "leaf_class", "lexical", "gap"),
                strict=True,
            )
        )
        context = (
            self.context_kind_embedding(batch.context_kind[:, :sequence_steps])
            + self.context_edge_embedding(batch.context_edge[:, :sequence_steps])
        )
        positions = torch.arange(sequence_steps, device=batch.action.device)
        target = (
            previous
            + action_history
            + context
            + self.state_projection(batch.state_features[:, :sequence_steps])
            + self.step_position_embedding(positions)[None]
        )
        target = target + self.start_embedding[None, None, :] * positions.eq(0)[
            None,
            :,
            None,
        ]
        causal = torch.triu(
            torch.ones(
                sequence_steps,
                sequence_steps,
                dtype=torch.bool,
                device=target.device,
            ),
            diagonal=1,
        )
        hidden = self.decoder_norm(
            self.decoder(
                target,
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=~batch.step_mask[:, :sequence_steps],
                memory_key_padding_mask=~batch.prompt_token_mask,
            )
        )
        node_context = (
            self.context_kind_embedding(batch.context_kind[:, :sequence_steps])
            + self.context_edge_embedding(batch.context_edge[:, :sequence_steps])
        )
        output = {
            name: head(hidden)
            for name, head in self.heads.items()
            if name != "node_kind"
        }
        output["node_kind"] = self.heads["node_kind"](hidden + node_context)
        candidate_state = self.candidate_projection(
            self.candidate_node_embedding(
                batch.candidate_kind[:, :sequence_steps]
            )
            + self.candidate_scope_embedding(
                batch.candidate_scope[:, :sequence_steps]
            )
            + self.candidate_position_embedding(
                batch.candidate_position[:, :sequence_steps]
            )
        )
        output["parent"] = torch.einsum(
            "btd,btcd->btc",
            self.parent_query(hidden),
            candidate_state,
        ).masked_fill(
            ~batch.candidate_mask[:, :sequence_steps],
            -torch.inf,
        )
        copy_scores = torch.einsum(
            "btd,bld->btl",
            self.copy_query(hidden),
            char_states,
        )
        output["copy_start"] = copy_scores.masked_fill(
            ~batch.prompt_masks[:, None, :],
            -torch.inf,
        )
        output["node_budget"] = output["node_budget"][:, 0]
        output["pooled_prompt"] = pooled
        return output


def _move_batch(batch: PrefixPolicyBatch, device: torch.device) -> PrefixPolicyBatch:
    return PrefixPolicyBatch(
        **{
            name: value.to(device)
            for name, value in vars(batch).items()
            if isinstance(value, torch.Tensor)
        },
        history={
            name: value.to(device) for name, value in batch.history.items()
        },
        previous={
            name: value.to(device) for name, value in batch.previous.items()
        },
    )


def _slice_batch(batch: PrefixPolicyBatch, index: torch.Tensor) -> PrefixPolicyBatch:
    maximum = int(batch.step_mask[index].sum(dim=1).max())
    values = {
        name: value[index]
        for name, value in vars(batch).items()
        if isinstance(value, torch.Tensor)
        and name not in {"prompts", "prompt_masks", "prompt_token_ids", "prompt_token_mask", "node_budget"}
    }
    values = {
        name: value[:, :maximum] if value.ndim >= 2 else value
        for name, value in values.items()
    }
    return PrefixPolicyBatch(
        prompts=batch.prompts[index],
        prompt_masks=batch.prompt_masks[index],
        prompt_token_ids=batch.prompt_token_ids[index],
        prompt_token_mask=batch.prompt_token_mask[index],
        node_budget=batch.node_budget[index],
        history={
            name: value[index, :maximum]
            for name, value in batch.history.items()
        },
        previous={
            name: value[index, :maximum]
            for name, value in batch.previous.items()
        },
        **values,
    )


def _loss(
    output: Mapping[str, torch.Tensor],
    batch: PrefixPolicyBatch,
) -> torch.Tensor:
    active = batch.step_mask
    action = batch.action[active]
    loss = torch.nn.functional.cross_entropy(
        output["action"][active],
        action,
    )
    loss = loss + torch.nn.functional.cross_entropy(
        output["node_budget"],
        batch.node_budget,
    )
    open_mask = active & batch.action.eq(REAL_ACTIONS.index("open_node"))
    if open_mask.any():
        loss = loss + torch.nn.functional.cross_entropy(
            output["node_kind"][open_mask],
            batch.node_kind[open_mask],
        )
    attach_mask = active & batch.action.eq(REAL_ACTIONS.index("attach_edge"))
    if attach_mask.any():
        loss = (
            loss
            + torch.nn.functional.cross_entropy(
                output["edge_field"][attach_mask],
                batch.edge_field[attach_mask],
            )
            + torch.nn.functional.cross_entropy(
                output["parent"][attach_mask],
                batch.candidate_target[attach_mask],
            )
        )
    leaf_mask = active & batch.action.eq(REAL_ACTIONS.index("declare_leaf"))
    if leaf_mask.any():
        loss = (
            loss
            + torch.nn.functional.cross_entropy(
                output["leaf_node"][leaf_mask],
                batch.leaf_node[leaf_mask],
            )
            + torch.nn.functional.cross_entropy(
                output["leaf_class"][leaf_mask],
                batch.leaf_class[leaf_mask],
            )
            + torch.nn.functional.cross_entropy(
                output["lexical"][leaf_mask],
                batch.lexical[leaf_mask],
            )
            + torch.nn.functional.cross_entropy(
                output["gap"][leaf_mask],
                batch.gap[leaf_mask],
            )
            + torch.nn.functional.binary_cross_entropy_with_logits(
                output["copy"][leaf_mask].squeeze(-1),
                batch.copy_label[leaf_mask].float(),
            )
        )
        copy_mask = leaf_mask & batch.copy_label.bool()
        if copy_mask.any():
            loss = (
                loss
                + torch.nn.functional.cross_entropy(
                    output["copy_start"][copy_mask],
                    batch.copy_start[copy_mask],
                )
                + torch.nn.functional.cross_entropy(
                    output["copy_length"][copy_mask],
                    batch.copy_length[copy_mask],
                )
            )
    return loss


def train_prefix_tree_policy(
    train_batch: PrefixPolicyBatch,
    validation_batch: PrefixPolicyBatch,
    vocabulary: Mapping[str, object],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_rows: int,
    hidden_size: int,
    layers: int,
    heads: int,
) -> tuple[PrefixTreePolicy, dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_data = _move_batch(train_batch, device)
    validation_data = _move_batch(validation_batch, device)
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_steps=train_batch.action.shape[1],
    ).to(device)
    optimizer = ScheduleFreeAdamW(model.parameters())
    generator = torch.Generator().manual_seed(seed + 1)
    history = []
    for _epoch in range(epochs):
        model.train()
        optimizer.train()
        order = torch.randperm(train_data.action.shape[0], generator=generator)
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch_rows):
            index = order[start : start + batch_rows].to(device)
            batch = _slice_batch(train_data, index)
            output = model(batch)
            loss = _loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
            steps += len(index)
        history.append(total / max(1, steps))
    model.eval()
    optimizer.eval()
    metrics = evaluate_prefix_policy(
        model,
        validation_data,
        batch_rows=batch_rows,
    )
    metrics["loss_history"] = history
    metrics["train_rows"] = train_batch.action.shape[0]
    metrics["validation_rows"] = validation_batch.action.shape[0]
    return model, metrics


@torch.no_grad()
def evaluate_prefix_policy(
    model: PrefixTreePolicy,
    batch: PrefixPolicyBatch,
    *,
    batch_rows: int,
) -> dict[str, object]:
    correct = {name: 0 for name in ("action", "node_kind", "edge_field", "parent", "leaf_class")}
    total = {name: 0 for name in correct}
    budget_correct = 0
    budget_total = 0
    for start in range(0, batch.action.shape[0], batch_rows):
        index = torch.arange(
            start,
            min(start + batch_rows, batch.action.shape[0]),
            device=batch.action.device,
        )
        local = _slice_batch(batch, index)
        output = model(local)
        action = local.action
        checks = {
            "action": action,
            "node_kind": local.node_kind,
            "edge_field": local.edge_field,
            "parent": local.candidate_target,
            "leaf_class": local.leaf_class,
        }
        masks = {
            "action": local.step_mask,
            "node_kind": local.step_mask & action.eq(REAL_ACTIONS.index("open_node")),
            "edge_field": local.step_mask & action.eq(REAL_ACTIONS.index("attach_edge")),
            "parent": local.step_mask & action.eq(REAL_ACTIONS.index("attach_edge")),
            "leaf_class": local.step_mask & action.eq(REAL_ACTIONS.index("declare_leaf")),
        }
        for name, target in checks.items():
            active = masks[name]
            if not active.any():
                continue
            predicted = output[name][active].argmax(dim=-1)
            correct[name] += int(predicted.eq(target[active]).sum())
            total[name] += int(active.sum())
        budget_correct += int(
            output["node_budget"].argmax(dim=-1).eq(local.node_budget).sum()
        )
        budget_total += len(index)
    return {
        name: correct[name] / max(1, total[name])
        for name in correct
    } | {
        "node_budget_accuracy": budget_correct / max(1, budget_total),
        "target_ast_input": False,
        "repair_or_retry": False,
    }


__all__ = [
    "PrefixTreePolicy",
    "evaluate_prefix_policy",
    "train_prefix_tree_policy",
]
