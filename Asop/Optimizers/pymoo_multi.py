"""Multi-objective pymoo adapters extracted from the Phase-1 runner."""
from __future__ import annotations

import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.operators.sampling.rnd import FloatRandomSampling

from .base import BaseOptimizer
from .utils import (
    build_reference_directions_for_nsga3,
    build_reference_points_for_rnsga,
    build_weights_for_rnsga,
    create_multiobjective_problem,
    get_variable_type_info,
    hyper_float,
    hyper_int,
    hyper_text,
    hyper_value,
    str_to_bool,
    warn_if_not_continuous_for_moo,
)
from Reporting.results import make_multiobjective_results


class NSGA2Optimizer(BaseOptimizer):
    algorithm_name = 'NSGA-II'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'NSGA-II'
        evaluation_backend = str(config.get('evaluation_backend', 'python_com')).lower()
        list_variable_types = []
        for input_var in config['inputs']:
            variable_type = input_var.get('variable_type', 'Continuous')
            list_variable_types.append(variable_type)
        unique_list = list(set(list_variable_types))
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        pop_size = int(hyper.get('pop_size', 40))
        n_gen = int(hyper.get('n_gen', hyper.get('generations', 50)))
        seed = int(hyper.get('seed', 42))
        if evaluation_backend == 'csharp_jsonl':
            nsga_kwargs = {'pop_size': pop_size, 'eliminate_duplicates': True}
            if len(unique_list) > 1:
                nsga_kwargs['n_offsprings'] = int(hyper.get('n_offsprings', min(10, pop_size)))
            algorithm = NSGA2(**nsga_kwargs)
        elif len(unique_list) == 1 and unique_list[0].lower() == 'continuous':
            algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
        elif len(unique_list) == 1 and unique_list[0].lower() == 'discrete':
            algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
        elif len(unique_list) > 1:
            n_offsprings = int(hyper.get('n_offsprings', min(10, pop_size)))
            algorithm = NSGA2(pop_size=pop_size, n_offsprings=n_offsprings, sampling=MixedVariableSampling(), mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()), eliminate_duplicates=MixedVariableDuplicateElimination())
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        pareto_F_raw = problem.restore_raw_objectives(res.F)
        results = {'algorithm': config['algorithm'], 'problem_type': config['problem_type'], 'objective_names': problem.objective_names, 'pareto_F_minimized': res.F.tolist() if res.F is not None else [], 'pareto_F_raw': pareto_F_raw.tolist() if pareto_F_raw is not None else [], 'pareto_X': res.X.tolist() if res.X is not None else [], 'history': problem.history}
        return self.finalize(results, problem.history, multi=True)

class NSGA3Optimizer(BaseOptimizer):
    algorithm_name = 'NSGA-III'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'NSGA-III'
        evaluation_backend = str(config.get('evaluation_backend', 'python_com')).lower()
        list_variable_types = []
        for input_var in config['inputs']:
            variable_type = input_var.get('variable_type', 'Continuous')
            list_variable_types.append(variable_type)
        unique_list = list(set(list_variable_types))
        unique_list_lower = [v.lower() for v in unique_list]
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        n_gen = int(hyper.get('n_gen', hyper.get('generations', 50)))
        seed = int(hyper.get('seed', 42))
        ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)
        pop_size = len(ref_dirs)
        nsga3_kwargs = {'ref_dirs': ref_dirs, 'pop_size': pop_size, 'eliminate_duplicates': True}
        if len(unique_list) > 1:
            nsga3_kwargs['n_offsprings'] = int(hyper.get('n_offsprings', min(10, pop_size)))
        algorithm = NSGA3(**nsga3_kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        pareto_F_raw = problem.restore_raw_objectives(res.F) if res.F is not None else None
        results = {'algorithm': config['algorithm'], 'problem_type': config['problem_type'], 'objective_names': problem.objective_names, 'pareto_F_minimized': res.F.tolist() if res.F is not None else [], 'pareto_F_raw': pareto_F_raw.tolist() if pareto_F_raw is not None else [], 'pareto_X': res.X.tolist() if res.X is not None else [], 'ref_dirs': ref_dirs.tolist(), 'history': problem.history}
        return self.finalize(results, problem.history, multi=True)

class RNSGA2Optimizer(BaseOptimizer):
    algorithm_name = 'R-NSGA-II'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'R-NSGA-II'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        pop_size = int(hyper.get('pop_size', 40))
        n_gen = int(hyper.get('n_gen', hyper.get('generations', 50)))
        seed = int(hyper.get('seed', 42))
        ref_points = build_reference_points_for_rnsga(config, problem)
        weights = build_weights_for_rnsga(config, problem)
        kwargs = {'ref_points': ref_points, 'pop_size': pop_size, 'epsilon': hyper_float(hyper, 'epsilon', 0.001), 'normalization': hyper_text(hyper, 'normalization', 'front'), 'extreme_points_as_reference_points': str_to_bool(hyper.get('extreme_points_as_reference_points', False), default=False), 'eliminate_duplicates': True}
        if weights is not None:
            kwargs['weights'] = weights
        if mixed and str(config.get('evaluation_backend', '')).lower() != 'csharp_jsonl':
            n_offsprings = int(hyper.get('n_offsprings', min(10, pop_size)))
            kwargs.update({'n_offsprings': n_offsprings, 'sampling': MixedVariableSampling(), 'mating': MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()), 'eliminate_duplicates': MixedVariableDuplicateElimination()})
        algorithm = RNSGA2(**kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'ref_points': ref_points.tolist(), 'weights': weights.tolist() if weights is not None else None})
        return self.finalize(results, problem.history, multi=True)

class RNSGA3Optimizer(BaseOptimizer):
    algorithm_name = 'R-NSGA-III'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'R-NSGA-III'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        n_gen = int(hyper.get('n_gen', hyper.get('generations', 50)))
        seed = int(hyper.get('seed', 42))
        ref_points = build_reference_points_for_rnsga(config, problem)
        pop_per_ref_point = hyper_int(hyper, 'pop_per_ref_point', hyper_value(hyper, 'pop_size', 50))
        mu = hyper_float(hyper, 'mu', 0.05)
        kwargs = {'ref_points': ref_points, 'pop_per_ref_point': pop_per_ref_point, 'mu': mu, 'eliminate_duplicates': True}
        if mixed and str(config.get('evaluation_backend', '')).lower() != 'csharp_jsonl':
            n_offsprings = int(hyper.get('n_offsprings', min(10, pop_per_ref_point)))
            kwargs.update({'n_offsprings': n_offsprings, 'sampling': MixedVariableSampling(), 'mating': MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()), 'eliminate_duplicates': MixedVariableDuplicateElimination()})
        algorithm = RNSGA3(**kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        ref_dirs_used = None
        try:
            ref_dirs_used = res.algorithm.survival.ref_dirs
        except Exception:
            ref_dirs_used = None
        results = make_multiobjective_results(config, problem, res, extra={'ref_points': ref_points.tolist(), 'pop_per_ref_point': pop_per_ref_point, 'mu': mu, 'ref_dirs': ref_dirs_used.tolist() if ref_dirs_used is not None else []})
        return self.finalize(results, problem.history, multi=True)

class MOPSOCDOptimizer(BaseOptimizer):
    algorithm_name = 'MOPSO-CD'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'MOPSO-CD'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        if not only_continuous:
            print('WARNING: MOPSO-CD works best with continuous variables.If you use discrete variables, the C# Aspen bridge will round discrete/integer coordinates before evaluation.', flush=True)
        pop_size = int(hyper.get('pop_size', 40))
        n_gen = int(hyper.get('n_gen', hyper.get('generations', 50)))
        seed = int(hyper.get('seed', 42))
        algorithm = MOPSO_CD(pop_size=pop_size, w=float(hyper.get('w', 0.6)), c1=float(hyper.get('c1', 2.0)), c2=float(hyper.get('c2', 2.0)), max_velocity_rate=float(hyper.get('max_velocity_rate', 0.5)), archive_size=int(hyper.get('archive_size', max(2 * pop_size, 100))), sampling=FloatRandomSampling())
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'pop_size': pop_size, 'archive_size': int(hyper.get('archive_size', max(2 * pop_size, 100)))})
        return self.finalize(results, problem.history, multi=True)

class CMOPSOOptimizer(BaseOptimizer):
    algorithm_name = 'CMOPSO'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'CMOPSO'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        if not only_continuous:
            print('WARNING: CMOPSO works best with continuous variables. If you use discrete variables, the C# Aspen bridge will round discrete/integer coordinates before evaluation.', flush=True)
        pop_size = int(hyper.get('pop_size', 40))
        n_gen = int(hyper.get('n_gen', hyper.get('generations', 50)))
        seed = int(hyper.get('seed', 42))
        algorithm = CMOPSO(pop_size=pop_size, max_velocity_rate=float(hyper.get('max_velocity_rate', 0.2)), elite_size=int(hyper.get('elite_size', 10)), initial_velocity=str(hyper.get('initial_velocity', 'random')), mutation_rate=float(hyper.get('mutation_rate', 0.5)), sampling=FloatRandomSampling())
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'pop_size': pop_size, 'elite_size': int(hyper.get('elite_size', 10))})
        return self.finalize(results, problem.history, multi=True)

class UNSGA3Optimizer(BaseOptimizer):
    algorithm_name = 'U-NSGA-III'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'U-NSGA-III'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        n_gen = hyper_int(hyper, 'n_gen', hyper_value(hyper, 'generations', 50))
        seed = hyper_int(hyper, 'seed', 42)
        ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)
        pop_size = hyper_int(hyper, 'pop_size', len(ref_dirs))
        kwargs = {'ref_dirs': ref_dirs, 'pop_size': pop_size, 'eliminate_duplicates': str_to_bool(hyper_value(hyper, 'eliminate_duplicates', True), default=True)}
        if 'n_offsprings' in hyper and str(hyper.get('n_offsprings', '')).strip() != '':
            kwargs['n_offsprings'] = hyper_int(hyper, 'n_offsprings', min(10, pop_size))
        if mixed and str(config.get('evaluation_backend', '')).lower() != 'csharp_jsonl':
            kwargs.update({'sampling': MixedVariableSampling(), 'mating': MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()), 'eliminate_duplicates': MixedVariableDuplicateElimination()})
        algorithm = UNSGA3(**kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'ref_dirs': ref_dirs.tolist(), 'pop_size': pop_size})
        return self.finalize(results, problem.history, multi=True)

class MOEADOptimizer(BaseOptimizer):
    algorithm_name = 'MOEA/D'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'MOEA/D'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        warn_if_not_continuous_for_moo('MOEA/D', only_continuous)
        n_gen = hyper_int(hyper, 'n_gen', hyper_value(hyper, 'generations', 50))
        seed = hyper_int(hyper, 'seed', 42)
        ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)
        n_neighbors = hyper_int(hyper, 'n_neighbors', 20)
        prob_neighbor_mating = hyper_float(hyper, 'prob_neighbor_mating', 0.9)
        algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=n_neighbors, prob_neighbor_mating=prob_neighbor_mating, sampling=FloatRandomSampling())
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'ref_dirs': ref_dirs.tolist(), 'n_neighbors': n_neighbors, 'prob_neighbor_mating': prob_neighbor_mating})
        return self.finalize(results, problem.history, multi=True)

class AGEMOEAOptimizer(BaseOptimizer):
    algorithm_name = 'AGE-MOEA'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'AGE-MOEA'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        warn_if_not_continuous_for_moo('AGE-MOEA', only_continuous)
        pop_size = hyper_int(hyper, 'pop_size', 40)
        n_gen = hyper_int(hyper, 'n_gen', hyper_value(hyper, 'generations', 50))
        seed = hyper_int(hyper, 'seed', 42)
        kwargs = {'pop_size': pop_size, 'eliminate_duplicates': str_to_bool(hyper_value(hyper, 'eliminate_duplicates', True), default=True)}
        if 'n_offsprings' in hyper and str(hyper.get('n_offsprings', '')).strip() != '':
            kwargs['n_offsprings'] = hyper_int(hyper, 'n_offsprings', min(10, pop_size))
        algorithm = AGEMOEA(**kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'pop_size': pop_size})
        return self.finalize(results, problem.history, multi=True)

class AGEMOEA2Optimizer(BaseOptimizer):
    algorithm_name = 'AGE-MOEA2'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'AGE-MOEA2'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        warn_if_not_continuous_for_moo('AGE-MOEA2', only_continuous)
        pop_size = hyper_int(hyper, 'pop_size', 40)
        n_gen = hyper_int(hyper, 'n_gen', hyper_value(hyper, 'generations', 50))
        seed = hyper_int(hyper, 'seed', 42)
        kwargs = {'pop_size': pop_size, 'eliminate_duplicates': str_to_bool(hyper_value(hyper, 'eliminate_duplicates', True), default=True)}
        if 'n_offsprings' in hyper and str(hyper.get('n_offsprings', '')).strip() != '':
            kwargs['n_offsprings'] = hyper_int(hyper, 'n_offsprings', min(10, pop_size))
        algorithm = AGEMOEA2(**kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'pop_size': pop_size})
        return self.finalize(results, problem.history, multi=True)

class RVEAOptimizer(BaseOptimizer):
    algorithm_name = 'RVEA'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'RVEA'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        warn_if_not_continuous_for_moo('RVEA', only_continuous)
        n_gen = hyper_int(hyper, 'n_gen', hyper_value(hyper, 'generations', 50))
        seed = hyper_int(hyper, 'seed', 42)
        ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)
        pop_size = hyper_int(hyper, 'pop_size', len(ref_dirs))
        alpha = hyper_float(hyper, 'alpha', 2.0)
        adapt_freq = hyper_float(hyper, 'adapt_freq', 0.1)
        kwargs = {'ref_dirs': ref_dirs, 'alpha': alpha, 'adapt_freq': adapt_freq, 'pop_size': pop_size, 'eliminate_duplicates': str_to_bool(hyper_value(hyper, 'eliminate_duplicates', True), default=True)}
        if 'n_offsprings' in hyper and str(hyper.get('n_offsprings', '')).strip() != '':
            kwargs['n_offsprings'] = hyper_int(hyper, 'n_offsprings', min(10, pop_size))
        algorithm = RVEA(**kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'ref_dirs': ref_dirs.tolist(), 'pop_size': pop_size, 'alpha': alpha, 'adapt_freq': adapt_freq})
        return self.finalize(results, problem.history, multi=True)

class SMSEMOAOptimizer(BaseOptimizer):
    algorithm_name = 'SMS-EMOA'
    problem_type = 'multi_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'SMS-EMOA'
        problem, aspen = create_multiobjective_problem(config)
        hyper = config['hyperparameters']
        _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
        warn_if_not_continuous_for_moo('SMS-EMOA', only_continuous)
        pop_size = hyper_int(hyper, 'pop_size', 40)
        n_gen = hyper_int(hyper, 'n_gen', hyper_value(hyper, 'generations', 50))
        seed = hyper_int(hyper, 'seed', 42)
        normalize = str_to_bool(hyper_value(hyper, 'normalize', True), default=True)
        kwargs = {'pop_size': pop_size, 'normalize': normalize, 'eliminate_duplicates': str_to_bool(hyper_value(hyper, 'eliminate_duplicates', True), default=True)}
        if 'n_offsprings' in hyper and str(hyper.get('n_offsprings', '')).strip() != '':
            kwargs['n_offsprings'] = hyper_int(hyper, 'n_offsprings', 1)
        algorithm = SMSEMOA(**kwargs)
        res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False, save_history=False)
        results = make_multiobjective_results(config, problem, res, extra={'pop_size': pop_size, 'normalize': normalize})
        return self.finalize(results, problem.history, multi=True)
