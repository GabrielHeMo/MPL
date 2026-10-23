"""Common result normalization and serialization for MPL optimizers.

Phase 2 intentionally does not alter optimizer mathematics.  This module only
makes the result envelope consistent and centralizes JSON/CSV persistence.
Legacy result keys are retained for backward compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .csv_export import write_history_csv_mono, write_history_csv_multi


def normalize_problem_type(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def make_multiobjective_results(config, problem, res, extra=None):
    pareto_F_raw = problem.restore_raw_objectives(res.F) if res.F is not None else None
    results = {
        "algorithm": config["algorithm"],
        "problem_type": config["problem_type"],
        "objective_names": problem.objective_names,
        "pareto_F_minimized": res.F.tolist() if res.F is not None else [],
        "pareto_F_raw": pareto_F_raw.tolist() if pareto_F_raw is not None else [],
        "pareto_X": res.X.tolist() if res.X is not None else [],
        "history": problem.history,
    }
    if extra:
        results.update(extra)
    return results


def normalize_result(config: Mapping[str, Any], results: Mapping[str, Any] | None,
                     history: Sequence[Mapping[str, Any]] | None, *, multi: bool) -> dict[str, Any]:
    normalized = dict(results or {})
    history_list = list(history or [])

    normalized.setdefault("status", "success")
    normalized.setdefault("algorithm", config.get("algorithm", ""))
    normalized["problem_type"] = normalize_problem_type(
        normalized.get("problem_type", config.get("problem_type", ""))
    )
    # Common Phase-2 contract.  Keep legacy history as an alias for MOO callers.
    normalized["evaluations"] = history_list

    if multi:
        normalized.setdefault("pareto_X", [])
        normalized.setdefault("pareto_F_raw", [])
        normalized.setdefault("pareto_F_minimized", [])
        normalized.setdefault("objective_names", [])
        normalized.setdefault("history", history_list)
    else:
        normalized.setdefault("best_x", None)
        normalized.setdefault("best_objective", None)
        normalized.setdefault("outputs", None)

    return normalized


def save_results(results_path: str, results: Mapping[str, Any]) -> None:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(results), f, indent=4)


def finalize_run(config, results_path: str, results, history, *, multi: bool):
    normalized = normalize_result(config, results, history, multi=multi)
    save_results(results_path, normalized)
    if multi:
        write_history_csv_multi(config, results_path, history)
    else:
        write_history_csv_mono(config, results_path, history)
    return normalized
