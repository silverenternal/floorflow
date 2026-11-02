# NLP Flow Rust workspace

This workspace provides compile-time declarations and code generation for the
natural-language cross-domain route.

- `nlp_flow_core`: runtime traits and immutable hierarchy specifications.
- `nlp_flow_macros`: typed channel, hierarchy, benchmark-target, and output
  contract procedural macros.
- `nlp_flow_codegen`: renderer that keeps the Python `nlp_flow` constants in
  sync with the Rust declarations. The generated Python file now covers
  benchmark heads, training scopes, architecture-search axes, JSON actions,
  tree domains, output modes, and domain-specific validity requirements.

Typical usage:

```rust
use nlp_flow_core::{HierarchyLevels, TypedChannel};
use nlp_flow_macros::{HierarchyLevels, TypedChannel};

#[derive(TypedChannel)]
enum TextChannel {
    #[nlp_flow(name = "syntax")]
    Syntax,
    #[nlp_flow(name = "entity")]
    Entity,
}

#[derive(HierarchyLevels)]
enum NlpHierarchy {
    #[nlp_flow(id = "L0", role = "document_plan", max_state_tokens = 16)]
    DocumentPlan,
}
```

Generate and verify the Python contract:

```bash
cargo run \
  --manifest-path rust/Cargo.toml \
  -p nlp_flow_codegen -- \
  --format python \
  --output src/floorflow/nlp_flow/_generated_protocol.py

cargo run \
  --manifest-path rust/Cargo.toml \
  -p nlp_flow_codegen -- \
  --format python \
  --check src/floorflow/nlp_flow/_generated_protocol.py
```

Use the model-contract audit for normal iteration:

```bash
uv run python scripts/audit_nlp_rust_contracts.py --sync
```

`--sync` regenerates the Python contract before running the stale-file check.
Omit it in CI and pull requests.
