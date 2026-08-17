# Repository Guidelines

## Project Structure & Module Organization

`prmeval/` contains the evaluation package. Schemas, registries, configuration, orchestration, and artifact handling live in `prmeval/core/`; dataset adapters and Stage 1 sampling are in `prmeval/sample/`; remote model integrations and Stage 2 inference are in `prmeval/infer/`; and Stage 3 metrics are in `prmeval/metrics/`. The main CLI is `prmeval/cli.py`; dataset preprocessing also exposes `prmeval/sample/prepare.py`.

Built-in infer implementations live one model per file in `prmeval/infer/baselines/`, using the registry name as the filename (for example, `topreward` belongs in `baselines/topreward.py`). Shared baseline helpers belong in `baselines/common.py`; generic HTTP and OpenAI-compatible Chat Completions behavior belongs in `infer/base.py` and `infer/openai.py`. Import every built-in baseline from `baselines/__init__.py` so registration occurs when `prmeval.infer` is imported. `prmeval.infer.create_infer()` is the public construction entry point; do not recreate an `infer/adapters.py` forwarding layer.

`dataset_unify/` is a separate raw-dataset conversion tool. Its loaders are in `dataset_unify/dataset_loaders/`, converter registration is in `dataset_unify/converters.py`, conversion YAML files are in `dataset_unify/configs/data_gen_configs/`, and source-specific instructions are in `dataset_unify/dataset_guides/`. Do not make PRMEval import dataset-specific loaders directly.

Keep tests in `tests/`, with reusable inputs under `tests/fixtures/`. Evaluation YAML files belong in `configs/eval/`. Put runnable smoke examples and checked-in expected artifacts in `examples/`; maintain protocol and configuration documentation in `docs/`.

## Build, Test, and Development Commands

- `pip install -e '.[dev]'` installs the package in editable mode with pytest and Ruff.
- `pytest` runs the complete test suite.
- `ruff check .` checks imports, style, common bugs, and Python modernization rules.
- `ruff format --check .` verifies formatting; use `ruff format .` to apply it.
- `python -m prmeval.cli --help` lists evaluation commands.
- `python -m prmeval.cli sample --config configs/eval/test_stage.yaml` runs Stage 1; replace `sample` with `infer` or `metrics` for Stages 2 and 3.
- `python -m prmeval.cli run --config configs/eval/test_stage.yaml` runs all three stages; inference requires configured `BASE_URL`, `MODEL_ID`, and, when applicable, `OPENAI_API_KEY` environment variables.
- `python -m dataset_unify.generate_hf_dataset --config_path=<yaml>` converts a supported raw dataset to the standard local Hugging Face Dataset format.
- `python -m dataset_unify.validate_dataset <dataset_dir>` validates a converted dataset before evaluation.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.10+ syntax, type hints, and a 120-character line limit. Ruff enforces `E`, `W`, `F`, `I`, `B`, `C4`, `UP`, and `RUF` rules. Use `snake_case` for functions, modules, and variables; `PascalCase` for classes; and descriptive YAML names.

Preserve the `bench.record.v1` `EvaluationRecord` contract across sampled and inferred artifacts. New datasets, samplers, infer baselines, and metrics should use the registries in `prmeval/core/registry.py`. All built-in infer baselines use OpenAI-compatible `POST /v1/chat/completions`; do not add local model/checkpoint loading or revive the removed `specialized`/`v1/evaluations` transport. Dataset-unification converters should use the registry in `dataset_unify/registry.py` and produce the fixed schema through `build_standard_dataset()` rather than adding source-specific fields to PRMEval.

## Testing Guidelines

Tests use `unittest.TestCase` and are collected by pytest. Name files `test_*.py` and methods `test_<behavior>`. Add focused regression tests for schema validation, baselines, samplers, registries, OpenAI Chat request contracts, metric outputs, and dataset-unification contracts. Prefer mocks and small checked-in fixtures over live services. The dataset-unification contract test may generate a minimal local MP4 but must not depend on external datasets or services.

Run `pytest`, `ruff check .`, and `ruff format --check .` before submitting changes. For targeted iteration, use commands such as `pytest tests/test_evaluation.py` or `pytest tests/test_dataset_unify_contract.py`.

## Commit & Pull Request Guidelines

History is minimal and does not establish a strict commit format. Write concise, imperative subjects that describe one logical change, for example `Add retry coverage for remote inference`. Pull requests should explain motivation and behavior changes, list validation commands, and link relevant issues. Include sample output or screenshots when metrics or visualizations change, and call out schema or configuration compatibility impacts.

## Security & Configuration

Never commit API keys, credentials, private endpoints, generated datasets, decoded media, or `evaluation_output/`. Reference secrets by environment-variable name, such as `api_key: OPENAI_API_KEY`, and document any new required variables. The CLI does not load `.env` automatically.

Treat configuration and schema changes as compatibility-sensitive. Update the relevant files in `docs/`, example YAML, smoke artifacts, and regression tests together. Keep generated dataset paths local and ensure the unified `frames` field is relative to the saved Dataset root.
