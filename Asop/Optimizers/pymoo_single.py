"""Single-objective pymoo adapters extracted verbatim from the Phase-1 runner."""
from __future__ import annotations

import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.nrbo import NRBO
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from cmaes_setup import CMAES, initialize_CMAES
from .base import BaseOptimizer
from .utils import create_single_objective_problem


class PSOOptimizer(BaseOptimizer):
    algorithm_name = 'PSO'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'PSO'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        algorithm = PSO(pop_size=int(hyper['pop_size']), w=float(hyper['w']), c1=float(hyper['c1']), c2=float(hyper['c2']), adaptive=str(hyper['adaptive']).lower() == 'true', initial_velocity=hyper['initial_velocity'], max_velocity_rate=float(hyper['max_velocity_rate']), pertube_best=str(hyper['pertube_best']).lower() == 'true')
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'PSO terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'PSO optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class GAOptimizer(BaseOptimizer):
    algorithm_name = 'GA'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'GA'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        sampling = None
        selection = None
        if str(hyper['sampling']).lower() == 'real_random':
            sampling = FloatRandomSampling()
        if str(hyper['crossover']).lower() == 'sbx':
            crossover = SBX()
        if str(hyper['mutation']).lower() == 'pm':
            mutation = PM()
        algorithm = GA(pop_size=int(hyper['pop_size']), sampling=sampling, crossover=crossover, mutation=mutation, eliminate_duplicates=str(hyper['eliminate_duplicates']).lower() == 'true', n_offsprings=int(hyper['offsprings']))
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'GA terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'GA optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class DEOptimizer(BaseOptimizer):
    algorithm_name = 'DE'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'DE'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        algorithm = DE(pop_size=int(hyper['pop_size']), n_offsprings=int(hyper['n_offsprings']), sampling=FloatRandomSampling())
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'DE terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'DE optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class NelderMeadOptimizer(BaseOptimizer):
    algorithm_name = 'NelderMead'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'NelderMead'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        algorithm = NelderMead(initialize_simplex=float(hyper['simplex']))
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'NelderMead terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'NelderMead optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class PatternSearchOptimizer(BaseOptimizer):
    algorithm_name = 'PatternSearch'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'PatternSearch'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        algorithm = PatternSearch(init_delta=float(hyper['delta']), init_rho=float(hyper['rho']), step_size=float(hyper['step_size']))
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'PatternSearch terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'PatternSearch optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class BRKGAOptimizer(BaseOptimizer):
    algorithm_name = 'BRKGA'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'BRKGA'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        if str(hyper['sampling']).lower() == 'real_random':
            sampling = FloatRandomSampling()
        algorithm = BRKGA(n_elites=int(hyper['n_elites']), n_offsprings=int(hyper['n_offsprings']), n_mutants=int(hyper['n_mutants']), bias=float(hyper['crossover_bias']), sampling=sampling)
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'BRKGA terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'BRKGA optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class ESOptimizer(BaseOptimizer):
    algorithm_name = 'ES'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'ES'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        if str(hyper['sampling']).lower() == 'real_random':
            sampling = FloatRandomSampling()
        algorithm = ES(n_offsprings=int(hyper['n_offsprings']), pop_size=int(hyper['pop_size']), rule=float(hyper['rule']), phi=float(hyper['phi']), gamma=float(hyper['gamma']))
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'ES terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'ES optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class SRESOptimizer(BaseOptimizer):
    algorithm_name = 'SRES'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'SRES'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        algorithm = SRES(PF=float(hyper['pf']))
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'SRES terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'SRES optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class ISRESOptimizer(BaseOptimizer):
    algorithm_name = 'ISRES'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'ISRES'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        algorithm = ISRES(n_offsprings=int(hyper['n_offsprings']), rule=float(hyper['rule']), gamma=float(hyper['gamma']), alpha=float(hyper['alpha']))
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'ISRES terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'ISRESoptimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class CMAESOptimizer(BaseOptimizer):
    algorithm_name = 'CMAES'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'CMAES'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        x0_array, x0_records = initialize_CMAES(config=config, n_init=int(hyper['n_init']), seed=int(hyper['seed']))
        if x0_array is None or len(x0_array) == 0:
            raise RuntimeError('No se pudo generar ningún x0 factible para CMAES.')
        x0 = np.asarray(x0_array[0], dtype=float).reshape(-1)
        opts = {'verb_disp': 0, 'verbose': -9, 'verb_log': 0}
        if 'pop_size' in hyper and str(hyper['pop_size']).strip() != '':
            opts['popsize'] = int(hyper['pop_size'])
        algorithm = CMAES(x0=x0, sigma=float(hyper['sigma']), normalize=True, opts=opts)
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'CMAES terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'CMAES optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class G3PCXOptimizer(BaseOptimizer):
    algorithm_name = 'G3PCX'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'G3PCX'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        if str(hyper['sampling']).lower() == 'real_random':
            sampling = FloatRandomSampling()
        algorithm = G3PCX(pop_size=int(hyper['pop_size']), n_offsprings=int(hyper['n_offsprings']), n_parents=int(hyper['n_parents']), family_size=int(hyper['family_size']), sampling=sampling)
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'G3PCX terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'G3PCX optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)

class NRBOOptimizer(BaseOptimizer):
    algorithm_name = 'NRBO'
    problem_type = 'single_objective'

    def run(self):
        config = self.config
        results_path = self.results_path
        algorithm_name = 'NRBO'
        problem, aspen = create_single_objective_problem(config)
        hyper = config['hyperparameters']
        algorithm = NRBO(pop_size=int(hyper['pop_size']), deciding_factor=float(hyper['deciding_factor']), max_iteration=int(hyper['max_iteration']))
        res = minimize(problem, algorithm, ('n_gen', int(hyper['generations'])), seed=int(hyper['seed']), verbose=False)
        valid_records = [row for row in problem.history if row['feasible'] and row['aspen_converged'] and np.isfinite(row['objective'])]
        if len(valid_records) == 0:
            results = {'status': 'error', 'message': 'NRBO terminó, pero ningún punto factible produjo una simulación Aspen convergida.', 'best_x': None, 'best_objective': None, 'outputs': None, 'evaluations': problem.history}
        else:
            best_record = min(valid_records, key=lambda r: r['objective'])
            results = {'status': 'success', 'message': 'NRBO optimization finished.', 'best_x': best_record['x'], 'best_objective': float(best_record['objective']), 'outputs': best_record['outputs'], 'evaluations': problem.history}
        return self.finalize(results, problem.history, multi=False)
