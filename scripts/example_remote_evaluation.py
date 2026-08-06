"""Run the public Python evaluation API with a YAML configuration."""

from __future__ import annotations

import argparse
import json

from prmeval import EvalConfig, Evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    summary = Evaluator(EvalConfig.from_yaml(args.config)).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
