# Chip Atom and Proxy-Real Audit v1

Status date: 2026-11-01. CPU/synthetic evidence only.

## Purpose

This audit prevents structural proxies from being treated as SOTA metrics before
they preserve the candidate decisions of the canonical legalized evaluation.

## Proxy-real audit

Inputs:

- `outputs/reports/chip_m74_m48_portfolio_seed20_official100_v1.json`
- `outputs/reports/chip_m74_m48_portfolio_seed21_official100_v1.json`

The two independent portfolio reports provide six model/legalizer arms over the
same 100 cases. They are used as a first candidate-view substitute for six
training checkpoints and are explicitly not described as training checkpoints.

Generated artifacts:

- `outputs/reports/chip_proxy_measurements_v1.json`
- `outputs/reports/chip_atom_metric_registry_v1.json`
- `outputs/reports/chip_proxy_real_consistency_v1.json`

| Atom | Proxy | Real metric | Spearman | Pearson | Top-policy agreement | Verdict |
|---|---|---:|---:|---:|---:|---|
| A03 | raw b2b HPWL | legalized b2b gap | 0.242 | 0.036 | 0.0 | diagnostic only |
| A10 | raw p2b HPWL | legalized p2b gap | 0.129 | 0.161 | 0.0 | diagnostic only |
| A27 | outline compactness | legalized area gap | 0.667 | 0.921 | 0.0 | near screen, not eligible |
| A27 | absolute outline aspect | legalized rho MAE | -0.052 | -0.024 | 0.0 | reject as selector |
| A28 | legalizer displacement | legalized cost | 0.453 | 0.890 | 1.0 | diagnostic only |
| A29 | legalizer rung | legalized cost | 0.651 | 0.662 | 1.0 | diagnostic only |

All six candidate-view measurements have a successful observed degradation contrast: the
proxy and canonical metric both worsen for at least one candidate-arm contrast.
None passes the complete decision-consistency contract.

The M27 canonical refiner curve adds seven real checkpoints over 32 shared
cases. Its target-free metrics also fail as selectors:

| Atom | Proxy | Real metric | Spearman | Verdict |
|---|---|---:|---:|---|
| A33 | raw overlap pairs | legalized cost | 0.519 | diagnostic only |
| A33 | raw overlap pairs | legalized V_rel | 0.492 | diagnostic only |
| A29 | legalizer rung | legalized cost | 0.654 | diagnostic only |
| A34 | legalizer solves | legalized cost | 0.649 | diagnostic only |
| A34 | legalizer wall time | legalized cost | 0.463 | diagnostic only |
| A08 | raw grouping violations | legalized grouping | 0.253 | reject as selector |
| A09 | raw aspect violations | legalized cost | flat | reject as selector |
| A26 | raw boundary violations | legalized boundary | 0.487 | reject: false-fail 0.833 |

The practical conclusion is that no raw geometry or legalizer telemetry from
these measurements may select a checkpoint, portfolio arm or claim.
Ten of 37 atoms now have at least one measured proxy-real pair; none of the 14
measurements passes the full contract.

## Atom self-evaluation

All 37 atoms are registered in
`configs/chip_sota_metric_ownership_v1.json`. The first runtime pass in
`outputs/reports/chip_atom_self_eval_v1.json` covers all 37 atoms:

- 29 active after enabling the atom gates;
- 1 exact zero-gate identity branch with reachable parameter tensors;
- 0 measured atoms with a completely absent gradient;
- 7 parameter-free/runtime atoms pass their dedicated CPU probes: A31-A37.
- controlled disable is sensitive for 30/30 parameterized atoms; A16 and A29
  use dedicated model/direct-head probes.

A29 is inactive through the ordinary objective path because the legalizer
response output is not consumed during generation. A direct head probe confirms
that it is gradient-reachable and trainable.

The synthetic batch is two cases, eight blocks and 12 pins. It validates wiring,
not quality or cross-case generalization. Shuffle and equal-parameter
substitutions remain open.

## Typed channel cache

`src/floorflow/chip_flow/typed_channel_cache.py` stores the complete pre-fusion
contract:

- block/global/structural/constraint/port tokens;
- all masks and port incidence;
- allowed-edge masks and group sizes;
- relation-specific memories and masks;
- diagnostics;
- encoder-state and input SHA-256 fingerprints.

Round-trip fusion is exactly equivalent in tests. Variable padded batch
collation is handled per case without falling back to online channel encoding.

`collate_typed_channel_states()` now pads variable port tokens, port masks,
incidence, allowed edges and relation-specific contracts. Cached single-case
states reproduce their individually fused outputs after collation.

The full frozen 5k subset is stored as 40 block-count shards. Maximum channel
error is `1.91e-6` and maximum fused error is `1.43e-6`, within the declared
`4e-6` tolerance. All 40 shard groups pass collation equivalence.

`prepare_hierarchy_context()` now consumes the cache when all pre-fusion typed
parameters are frozen. A fail-closed test replaces online encoding with an
exception and verifies that the cached path still constructs the hierarchy
context.

Sharded LRU loading reduces disk re-reads. On `cuda:0`, typed fusion-step medians
show `1.81x` speedup at batch 4 and `1.42x` at batch 16 with output error below
`2.9e-6`. This is not a full end-to-end training speed claim.

## Immediate next gates

1. Bind the remaining 27 chip atoms to six-snapshot or candidate-view data.
2. Add controlled disable, shuffle/zero sensitivity and equal-parameter
   substitutions for every atom.
3. Replace the present candidate-view proxy audit with six real training
   checkpoint evaluations before allowing proxy-based selection.
4. Use the surviving metric only within its demonstrated complexity and
   candidate family.
