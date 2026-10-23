import json
from pickle import FALSE, FLOAT
from pyclbr import Class
import sys
import time
import traceback
import numpy as np
import csv
import os 
import math
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf

import gpytorch
from .status import emit_status
from Evaluation.remote_aspen_problem_functions import CSharpAspenEvaluator
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

RUN_STATUS_DIR = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
FAILED_RUN_STATUS_VALUES = {9, 10}
PENALTY_VALUE = 1e10


def _emit_single_evaluation_status(algorithm_name, record):
    """High-frequency status update without permanently flooding the Activity Log."""
    objective = record.get("objective")
    try:
        objective_text = f"{float(objective):.6g}"
    except Exception:
        objective_text = str(objective)

    converged = bool(record.get("aspen_converged"))
    feasible = bool(record.get("feasible"))
    emit_status(
        f"{algorithm_name} evaluation {record.get('evaluation')} completed: "
        f"objective={objective_text}, converged={converged}, feasible={feasible}.",
        level="PROGRESS" if converged and feasible else "WARNING",
        stage="aspen_evaluation",
        current=record.get("evaluation"),
        algorithm=algorithm_name,
        log=False,
    )


class DiscreteBOBotorch:

    def __init__(self, config):
        self.config = config
        self.evaluator = CSharpAspenEvaluator(config)
        self.history_records = []

        self.hyper = config["hyperparameters"]

        self.n_init = int(self.hyper.get("n_init", 5))
        self.n_iter = int(self.hyper.get("n_iter", 30))
        self.beta = float(self.hyper.get("beta", 2.0))
        self.beta_h = float(self.hyper.get("beta_h", 25.0))

        self.lengthscale_min = float(self.hyper.get("lengthscale_min", 0.05))
        self.lengthscale_max = float(self.hyper.get("lengthscale_max", 2.0))
        self.lengthscale_trials = int(self.hyper.get("lengthscale_trials", 6))

        self.n_candidates = int(self.hyper.get("n_candidates", 2000))
        self.seed = int(self.hyper.get("seed", 42))

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.lb = np.array(
            [float(v["lower_bound"]) for v in config["inputs"]],
            dtype=float
        )

        self.ub = np.array(
            [float(v["upper_bound"]) for v in config["inputs"]],
            dtype=float
        )

        self.dim = len(self.lb)

        if np.any(self.ub <= self.lb):
            raise RuntimeError(
                "Error en bounds: todos los upper_bound deben ser mayores que lower_bound."
            )

    # ---------------------------------------------------------
    # Normalización
    # ---------------------------------------------------------
    def normalize_x(self, x):
        return (np.array(x, dtype=float) - self.lb) / (self.ub - self.lb)

    def denormalize_x(self, xn):
        return self.lb + np.array(xn, dtype=float) * (self.ub - self.lb)

    # ---------------------------------------------------------
    # Manejo de variables discretas
    # ---------------------------------------------------------
    def round_to_valid_x(self, x):
        x = np.array(x, dtype=float).reshape(-1)
        x = np.minimum(np.maximum(x, self.lb), self.ub)

        for i, input_var in enumerate(self.config["inputs"]):
            vtype = input_var.get("variable_type", "Continuous").lower()

            if vtype == "discrete":
                x[i] = round(x[i])
                x[i] = min(max(x[i], self.lb[i]), self.ub[i])

        return x.astype(float)

    def point_key(self, x):
        x = self.round_to_valid_x(x)
        key = []

        for i, input_var in enumerate(self.config["inputs"]):
            vtype = input_var.get("variable_type", "Continuous").lower()

            if vtype == "discrete":
                key.append(int(round(x[i])))
            else:
                key.append(round(float(x[i]), 10))

        return tuple(key)

    def preprocess_x(self, config, x):
        """
        Redondea variables discretas antes de enviarlas a Aspen.
        """
        x_processed = []

        for input_var, value in zip(config["inputs"], x):
            variable_type = input_var.get("variable_type", "Continuous")

            if variable_type.lower() == "discrete":
                value = round(value)

            x_processed.append(float(value))

        return x_processed
        
                
    def evaluate_point(self, config, x):
        """Evaluate one candidate through the C# -> Aspen JSONL bridge."""
        x = self.preprocess_x(config, x)
        try:
            return self.evaluator.evaluate_one(x)
        except Exception as e:
            return self.penalty_result(
                config=config,
                x=x,
                reason=f"Error requesting Aspen evaluation from C#: {e}",
            )
    def input_context(self, config, x):
        """
        Construye un diccionario tipo {'x1': valor, 'x2': valor, ...}
        para evaluar restricciones antes de correr Aspen.
        """
        ctx = {}
        for input_var, value in zip(config["inputs"], x):
            tag = input_var.get("tag") or input_var.get("alias")
            if tag:
                ctx[tag] = float(value)
        return ctx

    def normalize_key(self, d, *names, default=None):
        """
        Permite leer constraints aunque C# mande nombres como LeftSide, left_side,
        leftSide, RightSide, etc.
        """
        # print('d', d, names)
        if not isinstance(d, dict):
            return default
        lower_map = {str(k).lower().replace("_", ""): k for k in d.keys()}
        for name in names:
            key = lower_map.get(str(name).lower().replace("_", ""))
            if key is not None:
                return d[key]
        return default
    
    def resolve_value(self, token, ctx):
        """
        Convierte un lado de la restricción a número.
        Puede ser una constante, un tag tipo x1, o una expresión simple tipo x1 + x2.
        """
        # print('token', token , ctx)
        if isinstance(token, (int, float)):
            # print('aqui1', float(token))
            return float(token)

        text = str(token).strip()
        if text in ctx:
            # print('aqui2',float(ctx[text]))
            return float(ctx[text])

        try: # Retorna el valor numerico de una restricción 
            # print('aqui3', float(text))
            return float(text)
        except ValueError:
            pass

        safe_globals = {"__builtins__": {}}
        safe_locals = {
            **ctx,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "math": math,
            "np": np,
        }
        # print('float(eval(text, safe_globals, safe_locals))')
        return float(eval(text, safe_globals, safe_locals))
    
    def compare_values(self,left, operator, right, tol=1e-9):
        op = str(operator).strip()
        op = op.replace("≤", "<=").replace("≥", ">=").replace("==", "=")

        if op == "<":
            return left < right
        if op == "<=":
            return left <= right + tol
        if op == ">":
            return left > right
        if op == ">=":
            return left + tol >= right
        if op == "=":
            return abs(left - right) <= tol
        if op == "!=":
            return abs(left - right) > tol

        raise ValueError(f"Operador de restricción no soportado: {operator}")
    
    def check_input_constraints(self, config, x):
        """
        Evalúa restricciones de entrada antes de mandar el punto a Aspen.

        Formato esperado en config['constraints']:
        {
            'left_side' o 'LeftSide': 'x1',
            'operator' o 'Operator': '<=',
            'right_side' o 'RightSide': 'x2',
            'type' o 'Type': 'Hard'
        }

        También acepta expresiones como 'x4 + 1' < 'x6'.
        Solo filtra restricciones tipo Hard; las Soft se ignoran aquí.
        """
        constraints = config.get("constraints", []) or []
        ctx = self.input_context(config, x)
        violations = []

        for i, constraint in enumerate(constraints, start=1):
            ctype = str(self.normalize_key(constraint, "type", "constraint_type", default="Hard")).lower()
            if ctype == "soft":
                continue
            # Extra valores de izquierda y derecha de la restricción
            left_token = self.normalize_key(constraint, "left_side", "leftside", "left", default=None)
            operator = self.normalize_key(constraint, "operator", "op", default=None)
            right_token = self.normalize_key(constraint, "right_side", "rightside", "right", default=None)
        
            if left_token in [None, ""] or operator in [None, ""] or right_token in [None, ""]:
                continue

            try:
                #Evalua cada termino y compara valores con operadore para ver si cumple, si lo cumple, retorna True
                left_value = self.resolve_value(left_token, ctx)
                right_value = self.resolve_value(right_token, ctx)
           
                ok = self.compare_values(left_value, operator, right_value)
            except Exception as e:
                violations.append({
                    "constraint": i,
                    "left": str(left_token),
                    "operator": str(operator),
                    "right": str(right_token),
                    "message": f"No se pudo evaluar la restricción: {e}"
                })
                continue

            if not ok:
                violations.append({
                    "constraint": i,
                    "left": str(left_token),
                    "operator": str(operator),
                    "right": str(right_token),
                    "left_value": left_value,
                    "right_value": right_value,
                    "message": f"Restricción no cumplida: {left_token} {operator} {right_token}"
                })

        return len(violations) == 0, violations


    def satisfies_input_constraints(self, x):
        x = self.round_to_valid_x(x)
        ok, _ = self.check_input_constraints(self.config, x)
        return ok


    def evaluate_and_store(self, x):
        x = self.round_to_valid_x(x)
        result = self.evaluate_point(self.config, x)

        record = {
            "evaluation": len(self.history_records) + 1,
            "x": {
                self.config["inputs"][i]["tag"]: float(result["x"][i])
                for i in range(len(result["x"]))
            },
            "objective": float(result["objective"]),
            "outputs": result["outputs"],
            "feasible": bool(result["feasible"]),
            "aspen_converged": bool(result["aspen_converged"]),
            "run_status": result["run_status"],
            "message": result["message"],
            #"violations": result["violations"]
        }

        self.history_records.append(record)
        _emit_single_evaluation_status(str(self.config.get("algorithm", "Discrete Bayesian Optimization")), record)
        return record

    # ---------------------------------------------------------
    # Muestreo aleatorio evitando repetidos
    # ---------------------------------------------------------
    def random_unobserved(self, observed_keys, max_tries=5000):
        for _ in range(max_tries):
            xn = self.rng.random(self.dim)
            x = self.round_to_valid_x(self.denormalize_x(xn))

            key = self.point_key(x)

            if key in observed_keys:
                continue

            if not self.satisfies_input_constraints(x):
                continue

            return x

        return None

    # ---------------------------------------------------------
    # Modelo GP
    # ---------------------------------------------------------
    def build_model(self, train_X, train_Y):
        model = SingleTaskGP(
            train_X,
            train_Y,
            outcome_transform=Standardize(m=1),
            likelihood=GaussianLikelihood(
                noise_constraint=Interval(1e-6, 1e-1)
            ),
            covar_module=ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.dim,
                    lengthscale_constraint=Interval(0.01, 10.0)
                )
            )
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    def set_model_lengthscale(self, model, value):
        try:
            model.covar_module.base_kernel.lengthscale = float(value)
        except Exception:
            try:
                model.covar_module.lengthscale = float(value)
            except Exception:
                pass

    # ---------------------------------------------------------
    # Adquisición
    # ---------------------------------------------------------
    def acquisition_scores(self, model, train_Y, beta_value):
        bounds = torch.stack([
            torch.zeros(self.dim, dtype=torch.double),
            torch.ones(self.dim, dtype=torch.double)
        ])

        acqf = UpperConfidenceBound(
            model,
            beta=float(beta_value)
        )

        X_next, acq_value = optimize_acqf(
            acqf,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20
        )

        return X_next, acq_value

    def botorch_to_aspen_x(self, X_next):
        x_candidate_n = X_next.detach().cpu().numpy().reshape(-1)
        x_candidate = self.denormalize_x(x_candidate_n)
        x_candidate = self.round_to_valid_x(x_candidate)
        return x_candidate

    # ---------------------------------------------------------
    # Preparar datos para entrenar GP
    # ---------------------------------------------------------
    def get_valid_records_for_gp(self):
        valid_records = [
            r for r in self.history_records
            if r["feasible"]
            and r["aspen_converged"]
            and np.isfinite(r["objective"])
        ]

        return valid_records

    def build_training_data(self, valid_records):
        X_np = np.array([
            [r["x"][v["tag"]] for v in self.config["inputs"]]
            for r in valid_records
        ], dtype=float)

        # BoTorch maximiza.
        # Como tu problema Aspen minimiza, usamos Y = -objective.
        Y_np = np.array([
            [-float(r["objective"])]
            for r in valid_records
        ], dtype=float)

        train_X = torch.tensor(
            np.array([self.normalize_x(x) for x in X_np]),
            dtype=torch.double
        )

        train_Y = torch.tensor(
            Y_np,
            dtype=torch.double
        )

        return train_X, train_Y

    # ---------------------------------------------------------
    # Buscar alternativa si BoTorch propone repetido o inválido
    # ---------------------------------------------------------
    def find_alternative_candidate(
        self,
        model,
        train_Y,
        observed_keys,
        original_candidate
    ):
        best_alternative = None
        best_cost = np.inf

        beta_values = np.linspace(
            self.beta,
            self.beta_h,
            8
        )

        lengthscale_values = np.linspace(
            self.lengthscale_min,
            self.lengthscale_max,
            self.lengthscale_trials
        )

        original_xn = self.normalize_x(original_candidate)

        for beta_try in beta_values:
            for ls_try in lengthscale_values:

                self.set_model_lengthscale(model, ls_try)

                try:
                    X_try, _ = self.acquisition_scores(
                        model,
                        train_Y,
                        beta_try
                    )

                    x_try = self.botorch_to_aspen_x(X_try)

                except Exception as e:
                    emit_status(
                        f"Discrete BO alternative-candidate search failed for beta={beta_try}, "
                        f"lengthscale={ls_try}: {e}",
                        level="WARNING",
                        stage="acquisition",
                        algorithm="Discrete Bayesian Optimization",
                        log=False,
                    )
                    continue

                key_try = self.point_key(x_try)

                if key_try in observed_keys:
                    continue

                if not self.satisfies_input_constraints(x_try):
                    continue

                x_try_n = self.normalize_x(x_try)

                cost = (
                    abs(float(beta_try) - self.beta)
                    + np.sum((x_try_n - original_xn) ** 2)
                )

                if cost < best_cost:
                    best_cost = cost
                    best_alternative = x_try

        if best_alternative is None:
            best_alternative = self.random_unobserved(observed_keys)

        return best_alternative

    def penalty_result(self, config, x, reason, violations=None, run_status=None):
        x = self.preprocess_x(config, x)
        outputs = {output_var["tag"]: None for output_var in config.get("outputs", [])}
        return {
            "objective": PENALTY_VALUE,
            "outputs": outputs,
            "x": x,
            "feasible": False,
            "aspen_converged": False,
            "run_status": run_status,
            "message": reason,
            #"violations": violations or []
        }
 
    # ---------------------------------------------------------
    # Loop principal
    # ---------------------------------------------------------
    def run(self):
        observed_keys = set()
        total_steps = max(1, self.n_init + self.n_iter)

        emit_status(
            f"Starting Discrete Bayesian Optimization: n_init={self.n_init}, n_iter={self.n_iter}, "
            f"beta={self.beta}, candidates={self.n_candidates}, seed={self.seed}.",
            level="INFO",
            progress=0,
            stage="setup",
            algorithm="Discrete Bayesian Optimization",
        )

        # Diseño inicial
        while len(self.history_records) < self.n_init:
            x0 = self.random_unobserved(observed_keys)

            if x0 is None:
                emit_status(
                    "Discrete BO could not generate another valid initial-design point.",
                    level="WARNING",
                    stage="initial_design",
                    algorithm="Discrete Bayesian Optimization",
                )
                break

            observed_keys.add(self.point_key(x0))
            record = self.evaluate_and_store(x0)
            completed = len(self.history_records)
            emit_status(
                f"Discrete BO initial design {completed}/{self.n_init} completed. "
                f"Objective = {float(record['objective']):.6g}.",
                level="PROGRESS" if record.get("aspen_converged") else "WARNING",
                progress=100.0 * completed / total_steps,
                stage="initial_design",
                current=completed,
                total=self.n_init,
                algorithm="Discrete Bayesian Optimization",
            )

        # Optimización BO
        for iteration in range(self.n_iter):
            iteration_number = iteration + 1
            valid_records = self.get_valid_records_for_gp()

            emit_status(
                f"Discrete BO iteration {iteration_number}/{self.n_iter}: "
                f"{len(valid_records)} valid observations available for the GP.",
                level="INFO",
                stage="model",
                current=iteration_number,
                total=self.n_iter,
                algorithm="Discrete Bayesian Optimization",
            )

            if len(valid_records) < 2:
                emit_status(
                    "Not enough valid observations to fit the GP; using a random unexplored point.",
                    level="WARNING",
                    stage="candidate",
                    algorithm="Discrete Bayesian Optimization",
                )
                x_next = self.random_unobserved(observed_keys)

                if x_next is None:
                    break

                observed_keys.add(self.point_key(x_next))
                record = self.evaluate_and_store(x_next)
                completed = self.n_init + iteration_number
                emit_status(
                    f"Discrete BO iteration {iteration_number}/{self.n_iter} completed by fallback sampling. "
                    f"Objective = {float(record['objective']):.6g}.",
                    level="PROGRESS",
                    progress=100.0 * completed / total_steps,
                    stage="optimization",
                    current=iteration_number,
                    total=self.n_iter,
                    algorithm="Discrete Bayesian Optimization",
                )
                continue

            train_X, train_Y = self.build_training_data(valid_records)

            emit_status(
                f"Fitting Discrete BO Gaussian Process with {len(valid_records)} observations...",
                level="INFO",
                stage="gp_fit",
                algorithm="Discrete Bayesian Optimization",
                log=False,
            )
            model = self.build_model(train_X, train_Y)

            try:
                X_next, _ = self.acquisition_scores(
                    model,
                    train_Y,
                    self.beta
                )

                x_candidate = self.botorch_to_aspen_x(X_next)

            except Exception as e:
                emit_status(
                    f"Discrete BO acquisition optimization failed; using a random candidate: {e}",
                    level="WARNING",
                    stage="acquisition",
                    algorithm="Discrete Bayesian Optimization",
                )
                x_candidate = self.random_unobserved(observed_keys)

            if x_candidate is None:
                break

            candidate_key = self.point_key(x_candidate)

            needs_alternative = (
                candidate_key in observed_keys
                or not self.satisfies_input_constraints(x_candidate)
            )

            if needs_alternative:
                emit_status(
                    "Discrete BO acquisition proposed a repeated or invalid point; searching for an alternative.",
                    level="INFO",
                    stage="candidate",
                    algorithm="Discrete Bayesian Optimization",
                    log=False,
                )
                x_candidate = self.find_alternative_candidate(
                    model=model,
                    train_Y=train_Y,
                    observed_keys=observed_keys,
                    original_candidate=x_candidate
                )

                if x_candidate is None:
                    break

                candidate_key = self.point_key(x_candidate)

            observed_keys.add(candidate_key)
            record = self.evaluate_and_store(x_candidate)

            current_valid = self.get_valid_records_for_gp()
            best_value = min((r["objective"] for r in current_valid), default=record["objective"])
            completed = self.n_init + iteration_number
            emit_status(
                f"Discrete BO iteration {iteration_number}/{self.n_iter} completed. "
                f"Current best objective = {float(best_value):.6g}.",
                level="PROGRESS",
                progress=100.0 * completed / total_steps,
                stage="optimization",
                current=iteration_number,
                total=self.n_iter,
                algorithm="Discrete Bayesian Optimization",
            )

        # Selección final
        valid_records = self.get_valid_records_for_gp()

        if len(valid_records) == 0:
            emit_status(
                "Discrete Bayesian Optimization finished without any feasible Aspen-converged point.",
                level="ERROR",
                progress=100,
                stage="finished",
                algorithm="Discrete Bayesian Optimization",
            )
            return {
                "status": "error",
                "message": "Discrete BO terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "best_x": None,
                "best_objective": None,
                "outputs": None,
                "evaluations": self.history_records
            }, self.history_records

        best_record = min(
            valid_records,
            key=lambda r: r["objective"]
        )

        emit_status(
            f"Discrete Bayesian Optimization finished successfully. "
            f"Best objective = {float(best_record['objective']):.6g}.",
            level="SUCCESS",
            progress=100,
            stage="finished",
            algorithm="Discrete Bayesian Optimization",
        )

        return {
            "status": "success",
            "message": "Discrete Bayesian Optimization with BoTorch finished.",
            "best_x": best_record["x"],
            "best_objective": float(best_record["objective"]),
            "outputs": best_record["outputs"],
            "evaluations": self.history_records
        }, self.history_records


class VanillaBOBotorch:

    def __init__(self, config):
        self.history_records = []
        self.config = config
        self.hyper = config["hyperparameters"]
        self.evaluator = CSharpAspenEvaluator(config)

        self.n_init = int(self.hyper.get("n_init", 5))
        self.n_iter = int(self.hyper.get("n_iter", 30))
        self.n_candidates = int(self.hyper.get("n_candidates", 2000))
        self.seed = int(self.hyper.get("seed", 42))
        self.aqf = self.hyper.get("aqf", "Log-EI")

        self.beta = float(self.hyper.get("beta", 2.0))
        self.num_samples = int(self.hyper.get("num_samples", 128))

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.lb = np.array([float(v["lower_bound"]) for v in config["inputs"]], dtype=float)
        self.ub = np.array([float(v["upper_bound"]) for v in config["inputs"]], dtype=float)
        self.dim = len(self.lb)

        
    def normalize_x(self, x):
        return (np.array(x, dtype=float) - self.lb) / (self.ub - self.lb)

    def denormalize_x(self, xn):
        return self.lb + np.array(xn, dtype=float) * (self.ub - self.lb)

    def round_to_valid_x(self, x):
        x = np.array(x, dtype=float).reshape(-1)
        x = np.minimum(np.maximum(x, self.lb), self.ub)

        for i, input_var in enumerate(self.config["inputs"]):
            vtype = input_var.get("variable_type", "Continuous").lower()
            if vtype == "discrete":
                x[i] = round(x[i])
                x[i] = min(max(x[i], self.lb[i]), self.ub[i])

        return x.astype(float)

    # Restricciones
    def input_context(self, config, x):
        """
        Construye un diccionario tipo {'x1': valor, 'x2': valor, ...}
        para evaluar restricciones antes de correr Aspen.
        """
        ctx = {}
        for input_var, value in zip(config["inputs"], x):
            tag = input_var.get("tag") or input_var.get("alias")
            if tag:
                ctx[tag] = float(value)
        return ctx

    def normalize_key(self, d, *names, default=None):
        """
        Permite leer constraints aunque C# mande nombres como LeftSide, left_side,
        leftSide, RightSide, etc.
        """
        # print('d', d, names)
        if not isinstance(d, dict):
            return default
        lower_map = {str(k).lower().replace("_", ""): k for k in d.keys()}
        for name in names:
            key = lower_map.get(str(name).lower().replace("_", ""))
            if key is not None:
                return d[key]
        return default
    
    def resolve_value(self, token, ctx):
        """
        Convierte un lado de la restricción a número.
        Puede ser una constante, un tag tipo x1, o una expresión simple tipo x1 + x2.
        """
        # print('token', token , ctx)
        if isinstance(token, (int, float)):
            # print('aqui1', float(token))
            return float(token)

        text = str(token).strip()
        if text in ctx:
            # print('aqui2',float(ctx[text]))
            return float(ctx[text])

        try: # Retorna el valor numerico de una restricción 
            # print('aqui3', float(text))
            return float(text)
        except ValueError:
            pass

        safe_globals = {"__builtins__": {}}
        safe_locals = {
            **ctx,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "math": math,
            "np": np,
        }
        # print('float(eval(text, safe_globals, safe_locals))')
        return float(eval(text, safe_globals, safe_locals))
    
    def compare_values(self,left, operator, right, tol=1e-9):
        op = str(operator).strip()
        op = op.replace("≤", "<=").replace("≥", ">=").replace("==", "=")

        if op == "<":
            return left < right
        if op == "<=":
            return left <= right + tol
        if op == ">":
            return left > right
        if op == ">=":
            return left + tol >= right
        if op == "=":
            return abs(left - right) <= tol
        if op == "!=":
            return abs(left - right) > tol

        raise ValueError(f"Operador de restricción no soportado: {operator}")
    
    def check_input_constraints(self, config, x):
        """
        Evalúa restricciones de entrada antes de mandar el punto a Aspen.

        Formato esperado en config['constraints']:
        {
            'left_side' o 'LeftSide': 'x1',
            'operator' o 'Operator': '<=',
            'right_side' o 'RightSide': 'x2',
            'type' o 'Type': 'Hard'
        }

        También acepta expresiones como 'x4 + 1' < 'x6'.
        Solo filtra restricciones tipo Hard; las Soft se ignoran aquí.
        """
        constraints = config.get("constraints", []) or []
        ctx = self.input_context(config, x)
        violations = []

        for i, constraint in enumerate(constraints, start=1):
            ctype = str(self.normalize_key(constraint, "type", "constraint_type", default="Hard")).lower()
            if ctype == "soft":
                continue
            # Extra valores de izquierda y derecha de la restricción
            left_token = self.normalize_key(constraint, "left_side", "leftside", "left", default=None)
            operator = self.normalize_key(constraint, "operator", "op", default=None)
            right_token = self.normalize_key(constraint, "right_side", "rightside", "right", default=None)
        
            if left_token in [None, ""] or operator in [None, ""] or right_token in [None, ""]:
                continue

            try:
                #Evalua cada termino y compara valores con operadore para ver si cumple, si lo cumple, retorna True
                left_value = self.resolve_value(left_token, ctx)
                right_value = self.resolve_value(right_token, ctx)
           
                ok = self.compare_values(left_value, operator, right_value)
            except Exception as e:
                violations.append({
                    "constraint": i,
                    "left": str(left_token),
                    "operator": str(operator),
                    "right": str(right_token),
                    "message": f"No se pudo evaluar la restricción: {e}"
                })
                continue

            if not ok:
                violations.append({
                    "constraint": i,
                    "left": str(left_token),
                    "operator": str(operator),
                    "right": str(right_token),
                    "left_value": left_value,
                    "right_value": right_value,
                    "message": f"Restricción no cumplida: {left_token} {operator} {right_token}"
                })

        return len(violations) == 0, violations

    def penalty_result(self, config, x, reason, violations=None, run_status=None):
        x = self.preprocess_x(config, x)
        outputs = {output_var["tag"]: None for output_var in config.get("outputs", [])}
        return {
            "objective": PENALTY_VALUE,
            "outputs": outputs,
            "x": x,
            "feasible": False,
            "aspen_converged": False,
            "run_status": run_status,
            "message": reason,
            #"violations": violations or []
        }
 
                
    def satisfies_input_constraints(self, x):
        x = self.round_to_valid_x(x)
        ok, _ = self.check_input_constraints(self.config, x)
        return ok

    def point_key(self, x):
        x = self.round_to_valid_x(x)
        key = []

        for i, input_var in enumerate(self.config["inputs"]):
            vtype = input_var.get("variable_type", "Continuous").lower()

            if vtype == "discrete":
                key.append(int(round(x[i])))
            else:
                key.append(round(float(x[i]), 10))

        return tuple(key)


    def preprocess_x(self, config, x):
        """
        Redondea variables discretas antes de enviarlas a Aspen.
        """
        x_processed = []

        for input_var, value in zip(config["inputs"], x):
            variable_type = input_var.get("variable_type", "Continuous")

            if variable_type.lower() == "discrete":
                value = round(value)

            x_processed.append(float(value))

        return x_processed

    def evaluate_point(self, config, x):
        """Evaluate one candidate through the C# -> Aspen JSONL bridge."""
        x = self.preprocess_x(config, x)
        try:
            return self.evaluator.evaluate_one(x)
        except Exception as e:
            return self.penalty_result(
                config=config,
                x=x,
                reason=f"Error requesting Aspen evaluation from C#: {e}",
            )
    def evaluate_and_store(self, x):
        x = self.round_to_valid_x(x)
        result = self.evaluate_point(self.config, x)

        record = {
            "evaluation": len(self.history_records) + 1,
            "x": {
                self.config["inputs"][i]["tag"]: float(result["x"][i])
                for i in range(len(result["x"]))
            },
            "objective": float(result["objective"]),
            "outputs": result["outputs"],
            "feasible": bool(result["feasible"]),
            "aspen_converged": bool(result["aspen_converged"]),
            "run_status": result["run_status"],
            "message": result["message"],
            #"violations": result["violations"]
        }

        self.history_records.append(record)
        _emit_single_evaluation_status(str(self.config.get("algorithm", "Vanilla Bayesian Optimization")), record)
        return record

    def random_unobserved(self, observed_keys, max_tries=5000):
        for _ in range(max_tries):
            xn = self.rng.random(self.dim)
            x = self.round_to_valid_x(self.denormalize_x(xn))

            if self.point_key(x) in observed_keys:
                continue

            if not self.satisfies_input_constraints(x):
                continue

            return x

        return None

    def acquisition_scores(self, model, train_Y):
        aqf_clean = str(self.aqf).strip().lower()

        bounds = torch.stack([
            torch.zeros(self.dim, dtype=torch.double),
            torch.ones(self.dim, dtype=torch.double)
        ])

        if aqf_clean in ["log-ei", "logei", "log expected improvement"]:
            best_f = train_Y.max().item()
            acqf = LogExpectedImprovement(model, best_f=best_f)

        elif aqf_clean in ["upper confidence bound", "ucb"]:
            acqf = UpperConfidenceBound(model, beta=self.beta)

        else:
            raise RuntimeError(f"Función de adquisición no reconocida: {self.aqf}")

        X_next, acq_value = optimize_acqf(
            acqf,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=2000
        )

        return X_next, acq_value

    def build_model(self, train_X, train_Y):
        model = SingleTaskGP(
            train_X,
            train_Y,
            outcome_transform=Standardize(m=1),
            likelihood=GaussianLikelihood(
                noise_constraint=Interval(1e-6, 1e-1)
            ),
            covar_module=ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.dim,
                    lengthscale_constraint=Interval(0.01, 10.0)
                )
            )
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    def get_candidate_from_acquisition(self, model, train_Y):
        X_next, _ = self.acquisition_scores(model, train_Y)

        x_candidate_n = X_next.detach().cpu().numpy().reshape(-1)
        x_candidate = self.denormalize_x(x_candidate_n)
        x_candidate = self.round_to_valid_x(x_candidate)

        return x_candidate

    def local_unobserved_around_best(self, valid_records, observed_keys, max_tries=3000):
        best_record = min(valid_records, key=lambda r: r["objective"])

        x_best = np.array([
            best_record["x"][v["tag"]]
            for v in self.config["inputs"]
        ], dtype=float)

        x_best_n = self.normalize_x(x_best)

        radii = [0.10, 0.05, 0.02, 0.01, 0.005]
        tries_per_radius = max(1, max_tries // len(radii))

        for radius in radii:
            low = np.maximum(0.0, x_best_n - radius)
            high = np.minimum(1.0, x_best_n + radius)

            for _ in range(tries_per_radius):
                xn = self.rng.uniform(low, high)
                x = self.round_to_valid_x(self.denormalize_x(xn))
                key = self.point_key(x)

                if key in observed_keys:
                    continue

                if not self.satisfies_input_constraints(x):
                    continue

                return x

        return None

    def run(self):
        observed_keys = set()
        total_steps = max(1, self.n_init + self.n_iter)
        acquisition_name = str(self.aqf)

        emit_status(
            f"Starting Vanilla Bayesian Optimization: acquisition={acquisition_name}, "
            f"n_init={self.n_init}, n_iter={self.n_iter}, seed={self.seed}.",
            level="INFO",
            progress=0,
            stage="setup",
            algorithm="Vanilla Bayesian Optimization",
        )

        while len(self.history_records) < self.n_init:
            x0 = self.random_unobserved(observed_keys)

            if x0 is None:
                emit_status(
                    "Vanilla BO could not generate another valid initial-design point.",
                    level="WARNING",
                    stage="initial_design",
                    algorithm="Vanilla Bayesian Optimization",
                )
                break

            observed_keys.add(self.point_key(x0))
            record = self.evaluate_and_store(x0)
            completed = len(self.history_records)
            emit_status(
                f"Vanilla BO initial design {completed}/{self.n_init} completed. "
                f"Objective = {float(record['objective']):.6g}.",
                level="PROGRESS" if record.get("aspen_converged") else "WARNING",
                progress=100.0 * completed / total_steps,
                stage="initial_design",
                current=completed,
                total=self.n_init,
                algorithm="Vanilla Bayesian Optimization",
            )

        for iteration in range(self.n_iter):
            iteration_number = iteration + 1

            valid_records = [
                r for r in self.history_records
                if r["feasible"]
                and r["aspen_converged"]
                and np.isfinite(r["objective"])
            ]

            emit_status(
                f"Vanilla BO iteration {iteration_number}/{self.n_iter}: "
                f"{len(valid_records)} valid observations; acquisition={acquisition_name}.",
                level="INFO",
                stage="model",
                current=iteration_number,
                total=self.n_iter,
                algorithm="Vanilla Bayesian Optimization",
            )

            if len(valid_records) < 2:
                emit_status(
                    "Not enough valid observations to fit the GP; using a random unexplored point.",
                    level="WARNING",
                    stage="candidate",
                    algorithm="Vanilla Bayesian Optimization",
                )
                x_next = self.random_unobserved(observed_keys)

                if x_next is None:
                    break

                observed_keys.add(self.point_key(x_next))
                record = self.evaluate_and_store(x_next)
                completed = self.n_init + iteration_number
                emit_status(
                    f"Vanilla BO iteration {iteration_number}/{self.n_iter} completed by fallback sampling. "
                    f"Objective = {float(record['objective']):.6g}.",
                    level="PROGRESS",
                    progress=100.0 * completed / total_steps,
                    stage="optimization",
                    current=iteration_number,
                    total=self.n_iter,
                    algorithm="Vanilla Bayesian Optimization",
                )
                continue

            X_np = np.array([
                [r["x"][v["tag"]] for v in self.config["inputs"]]
                for r in valid_records
            ], dtype=float)

            Y_np = np.array([
                [-float(r["objective"])]
                for r in valid_records
            ], dtype=float)

            train_X = torch.tensor(
                np.array([self.normalize_x(x) for x in X_np]),
                dtype=torch.double
            )

            train_Y = torch.tensor(Y_np, dtype=torch.double)

            emit_status(
                f"Fitting Vanilla BO Gaussian Process with {len(valid_records)} observations...",
                level="INFO",
                stage="gp_fit",
                algorithm="Vanilla Bayesian Optimization",
                log=False,
            )
            model = self.build_model(train_X, train_Y)

            emit_status(
                f"Optimizing {acquisition_name} acquisition function...",
                level="INFO",
                stage="acquisition",
                algorithm="Vanilla Bayesian Optimization",
                log=False,
            )
            x_candidate = self.get_candidate_from_acquisition(model, train_Y)
            candidate_key = self.point_key(x_candidate)

            if candidate_key in observed_keys or not self.satisfies_input_constraints(x_candidate):
                emit_status(
                    "Vanilla BO acquisition proposed a repeated or invalid point; searching locally around the best point.",
                    level="INFO",
                    stage="candidate",
                    algorithm="Vanilla Bayesian Optimization",
                    log=False,
                )
                x_candidate = self.local_unobserved_around_best(
                    valid_records=valid_records,
                    observed_keys=observed_keys
                )

            if x_candidate is None:
                emit_status(
                    "Local candidate search failed; using a random unexplored point.",
                    level="WARNING",
                    stage="candidate",
                    algorithm="Vanilla Bayesian Optimization",
                )
                x_candidate = self.random_unobserved(observed_keys)

            if x_candidate is None:
                break

            candidate_key = self.point_key(x_candidate)
            observed_keys.add(candidate_key)
            self.evaluate_and_store(x_candidate)

            current_valid = [
                r for r in self.history_records
                if r["feasible"] and r["aspen_converged"] and np.isfinite(r["objective"])
            ]
            best_value = min((r["objective"] for r in current_valid), default=float("inf"))
            completed = self.n_init + iteration_number
            emit_status(
                f"Vanilla BO iteration {iteration_number}/{self.n_iter} completed. "
                f"Current best objective = {float(best_value):.6g}.",
                level="PROGRESS",
                progress=100.0 * completed / total_steps,
                stage="optimization",
                current=iteration_number,
                total=self.n_iter,
                algorithm="Vanilla Bayesian Optimization",
            )

        valid_records = [
            row for row in self.history_records
            if row["feasible"]
            and row["aspen_converged"]
            and np.isfinite(row["objective"])
        ]

        if len(valid_records) == 0:
            emit_status(
                "Vanilla Bayesian Optimization finished without any feasible Aspen-converged point.",
                level="ERROR",
                progress=100,
                stage="finished",
                algorithm="Vanilla Bayesian Optimization",
            )
            return {
                "status": "error",
                "message": "Vanilla BO terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "best_x": None,
                "best_objective": None,
                "outputs": None,
                "evaluations": self.history_records
            }, self.history_records

        best_record = min(valid_records, key=lambda r: r["objective"])

        emit_status(
            f"Vanilla Bayesian Optimization finished successfully using {acquisition_name}. "
            f"Best objective = {float(best_record['objective']):.6g}.",
            level="SUCCESS",
            progress=100,
            stage="finished",
            algorithm="Vanilla Bayesian Optimization",
        )

        return {
            "status": "success",
            "message": f"Vanilla Bayesian Optimization with BoTorch finished using {self.aqf}.",
            "best_x": best_record["x"],
            "best_objective": float(best_record["objective"]),
            "outputs": best_record["outputs"],
            "evaluations": self.history_records
        }, self.history_records

