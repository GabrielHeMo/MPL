"""Adapters for MPL custom, Bayesian and trust-region optimizers."""
from __future__ import annotations

from Protocol.status import emit_status
from .base import BaseOptimizer
from .utils import get_variable_type_info, require_csharp_backend
from .sboa_class import SBOA
from .bayesian_classes import DiscreteBOBotorch, VanillaBOBotorch
from .turbo_class import TrustRegionBOBotorch, ContinuousMOBOBotorch


class SBOAOptimizer(BaseOptimizer):
    algorithm_name = 'SBOA'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'SBOA'
        emit_status('Preparing SBOA optimization.', level='INFO', stage='setup', algorithm=algorithm_name)
        require_csharp_backend(config)
        optimizer = SBOA(config)
        results, history = optimizer.run()
        return self.finalize(results, history, multi=False)

class DiscreteBOOptimizer(BaseOptimizer):
    algorithm_name = 'Discrete Bayesian Optimization'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'Discrete Bayesian Optimization'
        emit_status('Preparing Discrete Bayesian Optimization.', level='INFO', stage='setup', algorithm=algorithm_name)
        require_csharp_backend(config)
        optimizer = DiscreteBOBotorch(config)
        results, history = optimizer.run()
        return self.finalize(results, history, multi=False)

class VanillaBOOptimizer(BaseOptimizer):
    algorithm_name = 'Vanilla Bayesian Optimization'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'Vanilla Bayesian Optimization'
        emit_status('Preparing Vanilla Bayesian Optimization.', level='INFO', stage='setup', algorithm=algorithm_name)
        require_csharp_backend(config)
        optimizer = VanillaBOBotorch(config)
        results, history = optimizer.run()
        return self.finalize(results, history, multi=False)

class TurboOptimizer(BaseOptimizer):
    algorithm_name = 'Trust Region Bayesian Optimization'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'Trust Region Bayesian Optimization'
        emit_status('Preparing Trust Region Bayesian Optimization.', level='INFO', stage='setup', algorithm=algorithm_name)
        _, _, only_continuous, _, _ = get_variable_type_info(config)
        if not only_continuous:
            raise RuntimeError('Trust Region Bayesian Optimization solo está habilitado para variables continuas. Cambia todas las variables de entrada a Continuous.')
        require_csharp_backend(config)
        optimizer = TrustRegionBOBotorch(config)
        results, history = optimizer.run()
        return self.finalize(results, history, multi=False)

class MorphBODisabledOptimizer(BaseOptimizer):
    algorithm_name = 'MORPHBO'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'MORPHBO'
        raise RuntimeError('MORPHBO is currently disabled because its optimizer class has not yet been migrated to the C# -> Aspen JSONL backend.')

class VanillaMOBOOptimizer(BaseOptimizer):
    algorithm_name = 'Vanilla M.O. Bayesian Optimization'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'Vanilla M.O. Bayesian Optimization'
        emit_status('Preparing continuous multi-objective Bayesian Optimization.', level='INFO', stage='setup', algorithm=algorithm_name)
        _, _, only_continuous, _, _ = get_variable_type_info(config)
        if not only_continuous:
            raise RuntimeError('Vanilla M.O. Bayesian Optimization solo está habilitado para variables continuas. Cambia todas las variables de entrada a Continuous.')
        require_csharp_backend(config)
        optimizer = ContinuousMOBOBotorch(config)
        results, history = optimizer.run()
        return self.finalize(results, history, multi=True)

class MorphMODisabledOptimizer(BaseOptimizer):
    algorithm_name = 'MORPHBO'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'MORPHBO'
        raise RuntimeError('MORPHBO is currently disabled because its optimizer class has not yet been migrated to the C# -> Aspen JSONL backend.')
