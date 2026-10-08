#!/usr/bin/env python3
"""CLI entry point for the reproducible evaluation."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from week4.evaluator import DEFAULT_MODELS, Week4Evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1")
    parser.add_argument("--output", type=Path, default=Path("data/evaluations"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()
    report = Week4Evaluator(
        args.base_url,
        args.output,
        models=args.models,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    ).run()
    print(f"Week 4 complete: {report['evaluation_runs']} runs")
    for model, metrics in report["aggregates"].items():
        print(
            f"{model}: accuracy={metrics['accuracy']:.1%}, "
            f"hallucination={metrics['hallucination_rate']:.1%}, "
            f"latency={metrics['latency_mean_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
