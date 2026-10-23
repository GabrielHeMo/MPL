"""Common optimizer interface used by OptimizerRegistry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from Reporting.results import finalize_run


class BaseOptimizer(ABC):
    algorithm_name: str = ""
    problem_type: str = ""

    def __init__(self, config: dict[str, Any], results_path: str):
        self.config = config
        self.results_path = results_path

    @abstractmethod
    def run(self) -> dict[str, Any]:
        raise NotImplementedError

    def finalize(self, results, history, *, multi: bool | None = None):
        if multi is None:
            multi = self.problem_type == "multi_objective"
        return finalize_run(
            self.config,
            self.results_path,
            results,
            history,
            multi=bool(multi),
        )
