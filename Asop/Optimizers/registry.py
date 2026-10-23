"""Central optimizer registry for MPL Phase 2."""
from __future__ import annotations

from typing import Type

from .base import BaseOptimizer
from .pymoo_single import (
    PSOOptimizer, GAOptimizer, DEOptimizer, NelderMeadOptimizer, PatternSearchOptimizer,
    BRKGAOptimizer, ESOptimizer, SRESOptimizer, ISRESOptimizer, CMAESOptimizer,
    G3PCXOptimizer, NRBOOptimizer,
)
from .pymoo_multi import (
    NSGA2Optimizer, NSGA3Optimizer, RNSGA2Optimizer, RNSGA3Optimizer, MOPSOCDOptimizer,
    CMOPSOOptimizer, UNSGA3Optimizer, MOEADOptimizer, AGEMOEAOptimizer,
    AGEMOEA2Optimizer, RVEAOptimizer, SMSEMOAOptimizer,
)
from .custom import (
    SBOAOptimizer, DiscreteBOOptimizer, VanillaBOOptimizer, TurboOptimizer,
    MorphBODisabledOptimizer, VanillaMOBOOptimizer, MorphMODisabledOptimizer,
)


def normalize_problem_type(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


class OptimizerRegistry:
    _registry: dict[tuple[str, str], Type[BaseOptimizer]] = {
        ("single_objective", "PSO"): PSOOptimizer,
        ("single_objective", "GA"): GAOptimizer,
        ("single_objective", "DE"): DEOptimizer,
        ("single_objective", "NelderMead"): NelderMeadOptimizer,
        ("single_objective", "PatternSearch"): PatternSearchOptimizer,
        ("single_objective", "BRKGA"): BRKGAOptimizer,
        ("single_objective", "ES"): ESOptimizer,
        ("single_objective", "SRES"): SRESOptimizer,
        ("single_objective", "ISRES"): ISRESOptimizer,
        ("single_objective", "CMAES"): CMAESOptimizer,
        ("single_objective", "G3PCX"): G3PCXOptimizer,
        ("single_objective", "NRBO"): NRBOOptimizer,
        ("single_objective", "SBOA"): SBOAOptimizer,
        ("single_objective", "Discrete Bayesian Optimization"): DiscreteBOOptimizer,
        ("single_objective", "Vanilla Bayesian Optimization"): VanillaBOOptimizer,
        ("single_objective", "Trust Region Bayesian Optimization"): TurboOptimizer,
        ("single_objective", "MORPHBO"): MorphBODisabledOptimizer,
        ("multi_objective", "NSGA-II"): NSGA2Optimizer,
        ("multi_objective", "NSGA-III"): NSGA3Optimizer,
        ("multi_objective", "R-NSGA-II"): RNSGA2Optimizer,
        ("multi_objective", "R-NSGA-III"): RNSGA3Optimizer,
        ("multi_objective", "MOPSO-CD"): MOPSOCDOptimizer,
        ("multi_objective", "CMOPSO"): CMOPSOOptimizer,
        ("multi_objective", "U-NSGA-III"): UNSGA3Optimizer,
        ("multi_objective", "MOEA/D"): MOEADOptimizer,
        ("multi_objective", "AGE-MOEA"): AGEMOEAOptimizer,
        ("multi_objective", "AGE-MOEA2"): AGEMOEA2Optimizer,
        ("multi_objective", "RVEA"): RVEAOptimizer,
        ("multi_objective", "SMS-EMOA"): SMSEMOAOptimizer,
        ("multi_objective", "Vanilla M.O. Bayesian Optimization"): VanillaMOBOOptimizer,
        ("multi_objective", "MORPHBO"): MorphMODisabledOptimizer,
    }

    @classmethod
    def create(cls, config: dict, results_path: str) -> BaseOptimizer:
        algorithm = str(config.get("algorithm", "")).strip()
        problem_type = normalize_problem_type(config.get("problem_type", ""))
        key = (problem_type, algorithm)
        optimizer_cls = cls._registry.get(key)
        if optimizer_cls is None:
            available = sorted(a for (p, a) in cls._registry if p == problem_type)
            raise ValueError(
                f"Unsupported optimizer '{algorithm}' for problem_type='{problem_type}'. "
                f"Available: {available}"
            )
        return optimizer_cls(config, results_path)

    @classmethod
    def registered_algorithms(cls, problem_type: str | None = None) -> list[str]:
        if problem_type is None:
            return sorted({a for _, a in cls._registry})
        normalized = normalize_problem_type(problem_type)
        return sorted(a for (p, a) in cls._registry if p == normalized)
