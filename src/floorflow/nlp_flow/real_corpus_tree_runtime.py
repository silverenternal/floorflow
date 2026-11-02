"""Target-free runtime for the real-corpus tree policy."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.ast_leaf_classifier import classify_leaf
from floorflow.nlp_flow.real_corpus_policy_data import (
    ACTION_HISTORY,
    MAX_NODES,
    MAX_PARENT_CANDIDATES,
    MAX_PROMPT,
    MAX_PROMPT_TOKENS,
    REAL_ACTIONS,
    previous_fields,
    prompt_token_ids,
    state_features,
)
from floorflow.nlp_flow.real_corpus_tree_policy import RealCorpusTreePolicy
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeAction,
    TreeActionDecoder,
    render_tree_structure,
)


def terminalizable_kinds(grammar: ASTTransitionGrammar) -> set[str]:
    terminal = {
        kind
        for kind, classes in grammar.leaf_classes.items()
        if classes
    }
    changed = True
    while changed:
        changed = False
        for transition in grammar.transitions:
            if (
                transition.parent_kind not in terminal
                and transition.child_kind in terminal
            ):
                terminal.add(transition.parent_kind)
                changed = True
    return terminal


def viable_parent_kinds(grammar: ASTTransitionGrammar) -> set[str]:
    terminal = terminalizable_kinds(grammar)
    return {
        transition.parent_kind
        for transition in grammar.transitions
        if transition.child_kind in terminal
    }


def _masked_choice(logits: torch.Tensor, allowed: set[int]) -> int:
    if not allowed:
        raise ValueError("no legal action candidate")
    masked = torch.full_like(logits, -torch.inf)
    for index in allowed:
        masked[index] = logits[index]
    return int(masked.argmax())


def _allowed_action_types(
    decoder: TreeActionDecoder,
    node_budget: int,
    parent_kinds: set[str],
    terminal_kinds: set[str],
) -> set[int]:
    if decoder.expected_scope is not None:
        return {REAL_ACTIONS.index("open_scope")}
    if not decoder.nodes:
        return {REAL_ACTIONS.index("open_node")}
    if decoder.pending_edge is not None:
        return {REAL_ACTIONS.index("open_node")}
    allowed = set()
    if len(decoder.nodes) < node_budget and any(
        bool(decoder.state.open_scopes)
        and node.scope_id == decoder.state.open_scopes[-1]
        and any(
            transition.parent_kind == node.kind
            and transition.child_kind in terminal_kinds
            for transition in decoder.grammar.transitions
        )
        for node in decoder.nodes
    ):
        allowed.add(REAL_ACTIONS.index("attach_edge"))
    if any(
        not node.children
        and node.leaf_class is None
        and bool(decoder.grammar.leaf_classes.get(node.kind))
        for node in decoder.nodes
    ):
        allowed.add(REAL_ACTIONS.index("declare_leaf"))
    active_scope = (
        decoder.state.open_scopes[-1]
        if decoder.state.open_scopes
        else None
    )
    scope_has_incomplete_node = any(
        node.scope_id == active_scope
        and not node.children
        and node.leaf_class is None
        for node in decoder.nodes
    )
    if decoder.state.open_scopes and not scope_has_incomplete_node:
        allowed.add(REAL_ACTIONS.index("close_scope"))
    undeclared = any(
        not node.children and node.leaf_class is None
        for node in decoder.nodes
    )
    if decoder.state.eos_ready and not undeclared:
        allowed.add(REAL_ACTIONS.index("terminate"))
    return allowed


def _previous_tensor(
    action: TreeAction | None,
    vocabulary: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = {}
    for name, value in previous_fields(action).items():
        mapping_name = {
            "action": "action_ids",
            "node": "node_ids",
            "edge": "edge_ids",
            "leaf_class": "class_ids",
            "lexical": "lexeme_ids",
            "gap": "gap_ids",
        }[name]
        output[name] = torch.tensor(
            [int(vocabulary[mapping_name].get(value, 0))],
            device=device,
        )
    return output


def _history_tensor(
    actions: Sequence[TreeAction],
    vocabulary: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = {}
    for name in ("action", "node", "edge", "leaf_class", "lexical", "gap"):
        mapping_name = {
            "action": "action_ids",
            "node": "node_ids",
            "edge": "edge_ids",
            "leaf_class": "class_ids",
            "lexical": "lexeme_ids",
            "gap": "gap_ids",
        }[name]
        values = [
            int(
                vocabulary[mapping_name].get(
                    previous_fields(action)[name],
                    0,
                )
            )
            for action in actions[-ACTION_HISTORY:]
        ]
        output[name] = torch.tensor(
            [[0] * (ACTION_HISTORY - len(values)) + values],
            device=device,
        )
    return output


def _inference_context(
    decoder: TreeActionDecoder,
    parent_kinds: set[str],
    vocabulary: Mapping[str, object],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
]:
    context_kind = "<none>"
    context_edge = "<none>"
    if decoder.pending_edge is not None:
        pending = decoder.pending_edge
        context_kind = decoder.nodes[pending.parent_index].kind
        context_edge = pending.edge_field
    elif not decoder.nodes:
        context_kind = "<start>"
    active_scope = (
        decoder.state.open_scopes[-1]
        if decoder.state.open_scopes
        else None
    )
    candidate_indices = [
        index
        for index, node in enumerate(decoder.nodes)
        if node.scope_id == active_scope
        and node.kind in parent_kinds
    ][-MAX_PARENT_CANDIDATES:]
    candidate_kind = [0] * MAX_PARENT_CANDIDATES
    candidate_scope = [0] * MAX_PARENT_CANDIDATES
    candidate_position = [0] * MAX_PARENT_CANDIDATES
    candidate_mask = [False] * MAX_PARENT_CANDIDATES
    scope_kinds = decoder.state.scope_kind_by_id
    for position, index in enumerate(candidate_indices):
        node = decoder.nodes[index]
        candidate_kind[position] = int(
            vocabulary["node_ids"].get(node.kind, 0)
        )
        candidate_scope[position] = int(
            vocabulary["scope_ids"].get(scope_kinds[node.scope_id], 0)
        )
        candidate_position[position] = min(index, MAX_NODES - 1)
        candidate_mask[position] = True
    return (
        torch.tensor(
            [int(vocabulary["node_ids"].get(context_kind, 0))],
            device=device,
        ),
        torch.tensor(
            [int(vocabulary["edge_ids"].get(context_edge, 0))],
            device=device,
        ),
        torch.tensor([candidate_kind], device=device),
        torch.tensor([candidate_scope], device=device),
        torch.tensor([candidate_position], device=device),
        torch.tensor([candidate_mask], dtype=torch.bool, device=device),
        candidate_indices,
    )


@torch.no_grad()
def generate_real_task(
    model: RealCorpusTreePolicy,
    prompt: str,
    language: str,
    grammar: ASTTransitionGrammar,
    vocabulary: Mapping[str, object],
    *,
    device: torch.device,
    max_steps: int = 4096,
) -> tuple[str | None, list[TreeAction], str | None]:
    token_ids = prompt_token_ids(prompt)
    char_ids = [
        int(vocabulary["chars"].get(character, 0))
        for character in prompt[:MAX_PROMPT]
    ]
    prompt_ids = torch.tensor(
        [token_ids + [0] * (MAX_PROMPT_TOKENS - len(token_ids))],
        device=device,
    )
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    prompt_chars = torch.tensor(
        [char_ids + [0] * (MAX_PROMPT - len(char_ids))],
        device=device,
    )
    prompt_char_mask = torch.tensor(
        [[True] * len(char_ids) + [False] * (MAX_PROMPT - len(char_ids))],
        dtype=torch.bool,
        device=device,
    )
    decoder = TreeActionDecoder(grammar)
    previous: TreeAction | None = None
    actions = []
    node_budget: int | None = None
    parent_kinds = viable_parent_kinds(grammar)
    terminal_kinds = terminalizable_kinds(grammar)
    try:
        for _step in range(max_steps):
            (
                context_kind,
                context_edge,
                candidate_kind,
                candidate_scope,
                candidate_position,
                candidate_mask,
                parent_candidates,
            ) = _inference_context(
                decoder,
                parent_kinds,
                vocabulary,
                device,
            )
            logits = model(
                prompt_ids,
                prompt_mask,
                prompt_chars,
                prompt_char_mask,
                torch.tensor(
                    [state_features(decoder)],
                    dtype=torch.float32,
                    device=device,
                ),
                context_kind,
                context_edge,
                candidate_kind,
                candidate_scope,
                candidate_position,
                candidate_mask,
                _history_tensor(actions, vocabulary, device),
                _previous_tensor(previous, vocabulary, device),
            )
            if node_budget is None:
                node_budget = max(
                    1,
                    int(logits["node_budget"][0].argmax()),
                )
            action = _select_real_action(
                decoder,
                grammar,
                logits,
                vocabulary,
                prompt,
                parent_candidates,
                node_budget,
                parent_kinds,
                terminal_kinds,
            )
            decoder.apply(action)
            actions.append(action)
            previous = action
            if isinstance(action, TerminateAction):
                source = render_tree_structure(decoder.structure())
                parse_ast_graph(source, language)
                return source, actions, None
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        return None, actions, str(error)
    return None, actions, "action generation exceeded the step cap"


def _select_real_action(
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
    logits: Mapping[str, torch.Tensor],
    vocabulary: Mapping[str, object],
    prompt: str,
    parent_candidates: Sequence[int],
    node_budget: int,
    parent_kinds: set[str],
    terminal_kinds: set[str],
) -> TreeAction:
    if decoder.expected_scope is not None:
        scope_id, node_index, scope_kind = decoder.expected_scope
        return OpenScopeAction(scope_id, node_index, scope_kind)
    action_id = _masked_choice(
        logits["action"][0],
        _allowed_action_types(decoder, node_budget, parent_kinds, terminal_kinds),
    )
    action_name = REAL_ACTIONS[action_id]
    if action_name == "open_node":
        if not decoder.nodes:
            allowed_nodes = {
                vocabulary["node_ids"][value]
                for value in grammar.legal_root_kinds
                if value in vocabulary["node_ids"]
                and (
                    value in terminal_kinds
                )
                and (
                    1 < node_budget
                    or bool(grammar.leaf_classes.get(value))
                )
            }
        else:
            pending = decoder.pending_edge
            if pending is None:
                raise ValueError("non-root node has no edge attachment")
            allowed_nodes = {
                vocabulary["node_ids"][value]
                for value in decoder.state.allowed_child_kinds(
                    parent_index=pending.parent_index,
                    edge_field=pending.edge_field,
                )
                if value in vocabulary["node_ids"]
                and (
                    value in terminal_kinds
                )
            }
        node_id = _masked_choice(logits["node_kind"][0], allowed_nodes)
        return OpenNodeAction(vocabulary["node_names"][node_id])
    if action_name == "attach_edge":
        parent_position = _masked_choice(
            logits["parent"][0],
            set(range(len(parent_candidates))),
        )
        parent = parent_candidates[parent_position]
        parent_kind = decoder.nodes[parent].kind
        allowed_edges = {
            vocabulary["edge_ids"][value.edge_field]
            for value in grammar.transitions
            if value.parent_kind == parent_kind
            and value.edge_field in vocabulary["edge_ids"]
            and value.child_kind in terminal_kinds
        }
        edge_id = _masked_choice(logits["edge_field"][0], allowed_edges)
        return AttachEdgeAction(parent, vocabulary["edge_names"][edge_id])
    if action_name == "declare_leaf":
        allowed_leaves = {
            index
            for index, node in enumerate(decoder.nodes)
            if not node.children
            and node.leaf_class is None
            and bool(grammar.leaf_classes.get(node.kind))
        }
        node_index = _masked_choice(logits["leaf_node"][0], allowed_leaves)
        node = decoder.nodes[node_index]
        allowed_classes = {
            vocabulary["class_ids"][value]
            for value in grammar.leaf_classes[node.kind]
            if value in vocabulary["class_ids"]
        }
        class_id = _masked_choice(logits["leaf_class"][0], allowed_classes)
        leaf_class = vocabulary["class_names"][class_id]
        if bool(torch.sigmoid(logits["copy"][0]).ge(0.5)):
            start = int(logits["copy_start"][0].argmax())
            length = int(logits["copy_length"][0].argmax())
            lexical = prompt[start : start + length]
            if classify_leaf(node.kind, lexical) != leaf_class:
                lexical = ""
        else:
            lexical = ""
        if not lexical:
            lexical_allowed = vocabulary["lexemes_by_kind_class"].get(
                f"{node.kind}|{leaf_class}",
                [],
            )
            if lexical_allowed:
                lexical_id = _masked_choice(
                    logits["lexical"][0],
                    set(lexical_allowed),
                )
                lexical = vocabulary["lexeme_names"][lexical_id]
            else:
                lexical = vocabulary["lexeme_names"][int(logits["lexical"][0].argmax())]
        gap_allowed = vocabulary["gaps_by_kind"].get(node.kind, [])
        gap_id = (
            _masked_choice(logits["gap"][0], set(gap_allowed))
            if gap_allowed
            else int(logits["gap"][0].argmax())
        )
        gap = vocabulary["gap_names"][gap_id]
        return DeclareLeafAction(
            node_index,
            leaf_class,
            lexical,
            gap,
        )
    if action_name == "close_scope":
        return CloseScopeAction(decoder.state.open_scopes[-1])
    return TerminateAction()


def save_policy_metadata(
    path,
    *,
    vocabulary: Mapping[str, object],
    metrics: Mapping[str, object],
) -> None:
    path.write_text(
        json.dumps({"vocabulary": vocabulary, "metrics": metrics}, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["generate_real_task", "save_policy_metadata"]
