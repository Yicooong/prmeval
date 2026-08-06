# Repository Guidelines

## Project Structure & Module Organization

`prmeval/` contains the Python package. Schemas and configuration live in `prmeval/core/`; dataset adapters and sampling are in `prmeval/data/`; remote model integrations are in `prmeval/baselines/`; orchestration and artifacts are in `prmeval/evaluation/`; and metrics are in `prmeval/metrics/`. CLI entry points are `prmeval/cli.py` and `prmeval/data/prepare.py`.

Keep tests in `tests/`, with reusable inputs under `tests/fixtures/`. Evaluation and preprocessing YAML files belong in `configs/eval/` and `configs/data/`. Put runnable demonstrations and their expected artifacts in `examples/`; maintain protocol documentation in `docs/`.

## Build, Test, and Development Commands

- `pip install -e '.[dev]'` installs the package in editable mode with pytest and Ruff.
- `pip install -e '.[viz]'` adds plotting support; dataset, video, and image dependencies are installed by default.
- `pytest` runs the complete test suite.
- `ruff check .` checks imports, style, common bugs, and Python modernization rules.
- `ruff format --check .` verifies formatting; use `ruff format .` to apply it.
- `python -m prmeval.cli --help` lists evaluation commands.
- `python -m prmeval.cli run --config configs/eval/test_stage.yaml` runs the three-stage smoke pipeline; inference requires a configured remote endpoint and API key.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.10+ syntax, type hints, and a 120-character line limit. Ruff enforces `E`, `W`, `F`, `I`, `B`, `C4`, `UP`, and `RUF` rules. Use `snake_case` for functions, modules, and variables; `PascalCase` for classes; and descriptive YAML names such as `full_smoke_jsonl.yaml`. Preserve the staged record contract (`sampled` to `inferred` to metrics) when changing schemas or artifacts.

## Testing Guidelines

Tests currently use `unittest.TestCase` and are collected by pytest. Name files `test_*.py` and methods `test_<behavior>`. Add focused regression tests for schema validation, adapters, samplers, remote request contracts, and metric outputs. Prefer mocks and small checked-in fixtures over live services. Run `pytest` and Ruff before submitting changes; no formal coverage threshold is configured.

## Commit & Pull Request Guidelines

History is minimal and does not establish a strict commit format. Write concise, imperative subjects that describe one logical change, for example `Add retry coverage for remote inference`. Pull requests should explain motivation and behavior changes, list validation commands, and link relevant issues. Include sample output or screenshots when metrics or visualizations change, and call out schema or configuration compatibility impacts.

## Security & Configuration

Never commit API keys, credentials, private endpoints, generated datasets, or `evaluation_output/`. Reference secrets by environment-variable name, such as `api_key: OPENAI_API_KEY`, and document any new required variables.
