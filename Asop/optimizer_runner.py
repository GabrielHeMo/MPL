"""MPL optimizer process entry point.

Phase 2 keeps this file intentionally small. Python still owns optimization,
objective expressions and input-side constraints; C# still exclusively owns
Aspen COM and physical simulation evaluation through the JSONL bridge.
"""
from __future__ import annotations

import json
import sys
import traceback

from Optimizers.registry import OptimizerRegistry
from Protocol.status import emit_status
from Reporting.results import save_results


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_problem_type(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: optimizer_runner.py <config_path> <results_path>")

    config_path = sys.argv[1]
    results_path = sys.argv[2]
    algorithm_name = None

    try:
        config = load_config(config_path)
        algorithm_name = str(config.get("algorithm", "")).strip()
        problem_type = normalize_problem_type(config.get("problem_type", ""))

        emit_status(
            f"Python optimizer runner started: {algorithm_name} ({problem_type}).",
            level="INFO",
            progress=0,
            stage="startup",
            algorithm=algorithm_name,
            log=True,
        )

        optimizer = OptimizerRegistry.create(config, results_path)
        optimizer.run()

    except Exception as exc:
        emit_status(
            f"Optimization failed: {exc}",
            level="ERROR",
            stage="error",
            algorithm=algorithm_name,
            log=True,
        )
        error_results = {
            "status": "error",
            "algorithm": algorithm_name or "",
            "problem_type": normalize_problem_type(locals().get("config", {}).get("problem_type", "")),
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "evaluations": [],
        }
        save_results(results_path, error_results)
        raise


if __name__ == "__main__":
    main()
