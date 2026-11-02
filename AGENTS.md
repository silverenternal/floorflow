# Repository Guidelines

## Project Structure & Module Organization

`src/floorflow/` contains the installable package. Its subpackages separate data and
taxonomy (`data/`), models and flow matching (`model/`, `flow/`), geometry and
representation, objectives, training, inference, and evaluation. Put executable
training, evaluation, and reporting workflows in `scripts/`. Keep tests in
`tests/test_<area>.py`, configuration snapshots in `configs/`, design notes in
`docs/`, and manuscript assets in `paper/`.

Datasets under `.data/` and generated files under `outputs/` are local only. Do not
commit datasets, reports, or model checkpoints; see `README.md` for the CubiCasa5K
data mount and workflow commands.

## Build, Test, and Development Commands

```bash
uv sync --extra dev --extra train  # Install project and development/training extras
uv build                           # Build source and wheel distributions
uv run pytest                      # Run the full test suite
uv run pytest tests/test_flow.py -q
uvx ruff check .                   # Lint Python sources
```

For CUDA training or the frozen evaluation pipeline, use the environment and
commands documented in `README.md`; those workflows depend on external data and
checkpoints.

## Coding Style & Naming Conventions

Use Python 3.11, four-space indentation, type hints for public interfaces, and a
120-character line limit. Ruff is the configured linter. Use `snake_case` for
modules, functions, and variables; `PascalCase` for classes; and
`UPPER_SNAKE_CASE` for constants. Name tests by observable behavior, for example
`test_flow_shapes_and_masking`.

Keep changes scoped to the relevant subpackage. Preserve explicit tensor-shape and
configuration semantics, and update adjacent documentation when behavior changes.

## Testing Guidelines

The suite uses pytest, with `src/` and `third_party/Trinity/src` added through
`pyproject.toml`. Add `test_*.py` modules and `test_*` functions near related
tests. Prefer deterministic, focused assertions on shapes, masks, gradients, and
metric behavior. Run the affected test file during development, then the full
suite before submitting. No coverage threshold is currently enforced.

## Commit & Pull Request Guidelines

This repository has no commit history yet. Use short, imperative subjects such as
`Add room support refiner`; include a body when motivation, data, or model behavior
needs explanation. Pull requests should describe the change, list verification
commands, and link relevant issues. For visual, model, or metric changes, identify
generated figures or reports and summarize the observed results. Keep generated
artifacts out of the pull request.

## Security & Configuration Tips

Keep credentials and machine-specific paths out of tracked files. Prefer the
configuration files in `configs/` for reproducible experiments, and record dataset
or model provenance in the appropriate file under `docs/`.
