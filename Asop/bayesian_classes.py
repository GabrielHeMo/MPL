import json
from pickle import FALSE, FLOAT
from pyclbr import Class
import sys
import time
import traceback
import numpy as np
import win32com.client
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
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

RUN_STATUS_DIR = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
FAILED_RUN_STATUS_VALUES = {9, 10}
PENALTY_VALUE = 1e10



class DiscreteBOBotorch:

    def __init__(self, config, aspen):
        self.config = config
        self.aspen = aspen
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
        
                
    def reinit_aspen(self, aspen):
        try:
            aspen.Reinit()
        except Exception:
            pass


    def evaluate_point(self, aspen, config, x):
        x = self.preprocess_x(config, x)
        # Primero se checa si se cumplen las restricciones
        ok_constraints, violations = self.check_input_constraints(config, x)

        if not ok_constraints:
            return self.penalty_result(
                config=config,
                x=x,
                reason="Punto rechazado antes de correr Aspen porque no cumple restricciones de entrada.",
                violations=violations
            )

        try:
            # Ahora añadimos los valores de restar a las variables para asegurar que si se actualizan no lance error aspen
            self.update_aspen(aspen , config)

            # Actualizar con valores reales sugeridos
            for input_var, value in zip(config["inputs"], x):
                self.set_aspen_variable(aspen, input_var["path"], value)
            # Se corre aspen
            self.run_aspen(aspen)
            # Checar estatus de convergencia de Aspen
            converged, run_status, status_message = self.check_aspen_convergence(aspen)
            if not converged:
                self.reinit_aspen(aspen)
                return self.penalty_result(
                    config=config,
                    x=x,
                    reason=status_message,
                    run_status=run_status
                )

            outputs = {}
            for output_var in config["outputs"]:
                outputs[output_var["tag"]] = self.get_aspen_variable(aspen, output_var["path"])

            obj = config["objectives"][0]
            expression = obj["expression"]

            safe_globals = {"__builtins__": {}}
            safe_locals = {
                **outputs,
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "math": math,
                "np": np,
            }
            objective_value = float(eval(expression, safe_globals, safe_locals))

            if obj["sense"].lower() == "maximize":
                objective_value = -objective_value

            if not np.isfinite(objective_value):
                raise RuntimeError(f"La función objetivo no es finita: {objective_value}")

            return {
                "objective": objective_value,
                "outputs": outputs,
                "x": x,
                "feasible": True,
                "aspen_converged": True,
                "run_status": run_status,
                "message": status_message,
                #"violations": []
            }

        except Exception as e:
            self.reinit_aspen(aspen)
            return self.penalty_result(
                config=config,
                x=x,
                reason=f"Error al evaluar el punto en Aspen: {e}"
            )


    # ---------------------------------------------------------
    # Restricciones
    # ---------------------------------------------------------
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


    def check_aspen_convergence(self, aspen):
        """
        Verifica si Aspen terminó correctamente usando el nodo Run-Status.

        En tu segundo código usabas UOSSTAT2 y penalizabas status 9 o 10.
        Aquí se conserva la misma idea, pero devolviendo (ok, status, message).
        """
        try:
            node = aspen.Tree.FindNode(RUN_STATUS_DIR)
        except Exception as e:
            return False, None, f"No se pudo leer Run-Status: {e}"

        if node is None:
            return False, None, f"No existe el nodo Run-Status: {RUN_STATUS_DIR}"

        try:
            status = int(node.Value)
        except Exception:
            return False, None, f"Run-Status no tiene un valor entero válido: {getattr(node, 'Value', None)}"

        if status in FAILED_RUN_STATUS_VALUES:
            return False, status, f"Aspen no convergió. Run-Status = {status}"

        return True, status, f"Aspen convergió. Run-Status = {status}"


    # ---------------------------------------------------------
    # Evaluación Aspen
    # ---------------------------------------------------------
    def evaluate_and_store(self, x):
        x = self.round_to_valid_x(x)
        result = self.evaluate_point(self.aspen, self.config, x)

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
                    print(
                        f"Error probando beta={beta_try}, "
                        f"lengthscale={ls_try}: {e}"
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

        # Diseño inicial
        while len(self.history_records) < self.n_init:
            x0 = self.random_unobserved(observed_keys)

            if x0 is None:
                break

            observed_keys.add(self.point_key(x0))
            self.evaluate_and_store(x0)

        # Optimización BO
        for iteration in range(self.n_iter):

            valid_records = self.get_valid_records_for_gp()

            if len(valid_records) < 2:
                x_next = self.random_unobserved(observed_keys)

                if x_next is None:
                    break

                observed_keys.add(self.point_key(x_next))
                self.evaluate_and_store(x_next)
                continue

            train_X, train_Y = self.build_training_data(valid_records)

            model = self.build_model(train_X, train_Y)

            try:
                X_next, _ = self.acquisition_scores(
                    model,
                    train_Y,
                    self.beta
                )

                x_candidate = self.botorch_to_aspen_x(X_next)

            except Exception as e:
                print(f"Error en optimize_acqf: {e}")
                x_candidate = self.random_unobserved(observed_keys)

            if x_candidate is None:
                break

            candidate_key = self.point_key(x_candidate)

            needs_alternative = (
                candidate_key in observed_keys
                or not self.satisfies_input_constraints(x_candidate)
            )

            if needs_alternative:
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
            self.evaluate_and_store(x_candidate)

        # Selección final
        valid_records = self.get_valid_records_for_gp()

        if len(valid_records) == 0:
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

        return {
            "status": "success",
            "message": "Discrete Bayesian Optimization with BoTorch finished.",
            "best_x": best_record["x"],
            "best_objective": float(best_record["objective"]),
            "outputs": best_record["outputs"],
            "evaluations": self.history_records
        }, self.history_records


    def set_aspen_variable(self,aspen, path, value):
        node = aspen.Tree.FindNode(path)

        if node is None:
            raise RuntimeError(f"The variable Aspen was not found.: {path}")

        node.Value = value


    def get_aspen_variable(self, aspen, path):
        node = aspen.Tree.FindNode(path)
        if node is None:
            raise RuntimeError(f"Aspen exit not found: {path}")
        if node.Value is None:
            raise RuntimeError(f"The Aspen output has no value: {path}")
        value = float(node.Value)
        if not np.isfinite(value):
            raise RuntimeError(f"The Aspen output is not finite: {path} = {value}")
        return value

    
    def update_aspen(self, aspen, config):
        """
        Actualiza Aspen con los restart_values antes de evaluar un nuevo punto.
        restart_values viene dentro de config["inputs"].
        """

        for input_var in config["inputs"]:
            path = input_var["path"]

            restart_value = input_var.get("restart_values", None)

            if restart_value is None or str(restart_value).strip() == "":
                restart_value = input_var.get("current_value", None)

            if restart_value is None or str(restart_value).strip() == "":
                raise RuntimeError(
                    f"No hay restart_value para {input_var.get('tag', path)}"
                )

            value = float(str(restart_value).replace(",", "."))

            variable_type = input_var.get("variable_type", "Continuous")

            if variable_type.lower() == "discrete":
                value = round(value)

            self.set_aspen_variable(aspen, path, value)

    def run_aspen(self, aspen):
        aspen.Run2()

        try:
            while aspen.Engine.IsRunning:
                time.sleep(0.2)
        except Exception:
            time.sleep(2)

                        
    def reinit_aspen(self, aspen):
        try:
            aspen.Reinit()
        except Exception:
            pass

 
class VanillaBOBotorch:

    def __init__(self, config, aspen):
        self.history_records = []
        self.config = config
        self.hyper = config["hyperparameters"]
        self.aspen = aspen

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

        
    def check_aspen_convergence(self, aspen):
        """
        Verifica si Aspen terminó correctamente usando el nodo Run-Status.

        En tu segundo código usabas UOSSTAT2 y penalizabas status 9 o 10.
        Aquí se conserva la misma idea, pero devolviendo (ok, status, message).
        """
        try:
            node = aspen.Tree.FindNode(RUN_STATUS_DIR)
        except Exception as e:
            return False, None, f"No se pudo leer Run-Status: {e}"

        if node is None:
            return False, None, f"No existe el nodo Run-Status: {RUN_STATUS_DIR}"

        try:
            status = int(node.Value)
        except Exception:
            return False, None, f"Run-Status no tiene un valor entero válido: {getattr(node, 'Value', None)}"

        if status in FAILED_RUN_STATUS_VALUES:
            return False, status, f"Aspen no convergió. Run-Status = {status}"

        return True, status, f"Aspen convergió. Run-Status = {status}"


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
 
                
    def reinit_aspen(self, aspen):
        try:
            aspen.Reinit()
        except Exception:
            pass


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

    def set_aspen_variable(self,aspen, path, value):
        node = aspen.Tree.FindNode(path)

        if node is None:
            raise RuntimeError(f"The variable Aspen was not found.: {path}")

        node.Value = value


    def get_aspen_variable(self, aspen, path):
        node = aspen.Tree.FindNode(path)
        if node is None:
            raise RuntimeError(f"Aspen exit not found: {path}")
        if node.Value is None:
            raise RuntimeError(f"The Aspen output has no value: {path}")
        value = float(node.Value)
        if not np.isfinite(value):
            raise RuntimeError(f"The Aspen output is not finite: {path} = {value}")
        return value

    
    def update_aspen(self, aspen, config):
        """
        Actualiza Aspen con los restart_values antes de evaluar un nuevo punto.
        restart_values viene dentro de config["inputs"].
        """

        for input_var in config["inputs"]:
            path = input_var["path"]

            restart_value = input_var.get("restart_values", None)

            if restart_value is None or str(restart_value).strip() == "":
                restart_value = input_var.get("current_value", None)

            if restart_value is None or str(restart_value).strip() == "":
                raise RuntimeError(
                    f"No hay restart_value para {input_var.get('tag', path)}"
                )

            value = float(str(restart_value).replace(",", "."))

            variable_type = input_var.get("variable_type", "Continuous")

            if variable_type.lower() == "discrete":
                value = round(value)

            self.set_aspen_variable(aspen, path, value)

    def run_aspen(self, aspen):
        aspen.Run2()

        try:
            while aspen.Engine.IsRunning:
                time.sleep(0.2)
        except Exception:
            time.sleep(2)

                        
    def reinit_aspen(self, aspen):
        try:
            aspen.Reinit()
        except Exception:
            pass


    def evaluate_point(self, aspen, config, x):
        x = self.preprocess_x(config, x)
        # Primero se checa si se cumplen las restricciones
        ok_constraints, violations = self.check_input_constraints(config, x)

        if not ok_constraints:
            return self.penalty_result(
                config=config,
                x=x,
                reason="Punto rechazado antes de correr Aspen porque no cumple restricciones de entrada.",
                violations=violations
            )

        try:
            # Ahora añadimos los valores de restar a las variables para asegurar que si se actualizan no lance error aspen
            self.update_aspen(aspen , config)

            # Actualizar con valores reales sugeridos
            for input_var, value in zip(config["inputs"], x):
                self.set_aspen_variable(aspen, input_var["path"], value)
            # Se corre aspen
            self.run_aspen(aspen)
            # Checar estatus de convergencia de Aspen
            converged, run_status, status_message = self.check_aspen_convergence(aspen)
            if not converged:
                self.reinit_aspen(aspen)
                return self.penalty_result(
                    config=config,
                    x=x,
                    reason=status_message,
                    run_status=run_status
                )

            outputs = {}
            for output_var in config["outputs"]:
                outputs[output_var["tag"]] = self.get_aspen_variable(aspen, output_var["path"])

            obj = config["objectives"][0]
            expression = obj["expression"]

            safe_globals = {"__builtins__": {}}
            safe_locals = {
                **outputs,
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "math": math,
                "np": np,
            }
            objective_value = float(eval(expression, safe_globals, safe_locals))

            if obj["sense"].lower() == "maximize":
                objective_value = -objective_value

            if not np.isfinite(objective_value):
                raise RuntimeError(f"La función objetivo no es finita: {objective_value}")

            return {
                "objective": objective_value,
                "outputs": outputs,
                "x": x,
                "feasible": True,
                "aspen_converged": True,
                "run_status": run_status,
                "message": status_message,
                #"violations": []
            }

        except Exception as e:
            self.reinit_aspen(aspen)
            return self.penalty_result(
                config=config,
                x=x,
                reason=f"Error al evaluar el punto en Aspen: {e}"
            )


    def evaluate_and_store(self, x):
        x = self.round_to_valid_x(x)
        result = self.evaluate_point(self.aspen, self.config, x)

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

        while len(self.history_records) < self.n_init:
            x0 = self.random_unobserved(observed_keys)

            if x0 is None:
                break

            observed_keys.add(self.point_key(x0))
            self.evaluate_and_store(x0)

        for iteration in range(self.n_iter):

            valid_records = [
                r for r in self.history_records
                if r["feasible"]
                and r["aspen_converged"]
                and np.isfinite(r["objective"])
            ]

            if len(valid_records) < 2:
                x_next = self.random_unobserved(observed_keys)

                if x_next is None:
                    break

                observed_keys.add(self.point_key(x_next))
                self.evaluate_and_store(x_next)
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

            model = self.build_model(train_X, train_Y)

            x_candidate = self.get_candidate_from_acquisition(model, train_Y)
            candidate_key = self.point_key(x_candidate)

            #if candidate_key in observed_keys or not self.satisfies_input_constraints(x_candidate):
            #    x_candidate = self.random_unobserved(observed_keys)

            if candidate_key in observed_keys or not self.satisfies_input_constraints(x_candidate):
                x_candidate = self.local_unobserved_around_best(
                    valid_records=valid_records,
                    observed_keys=observed_keys
                )

            if x_candidate is None:
                x_candidate = self.random_unobserved(observed_keys)

            candidate_key = self.point_key(x_candidate)
            observed_keys.add(candidate_key)
            self.evaluate_and_store(x_candidate)

        valid_records = [
            row for row in self.history_records
            if row["feasible"]
            and row["aspen_converged"]
            and np.isfinite(row["objective"])
        ]

        if len(valid_records) == 0:
            return {
                "status": "error",
                "message": "Vanilla BO terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "best_x": None,
                "best_objective": None,
                "outputs": None,
                "evaluations": self.history_records
            }, self.history_records

        best_record = min(valid_records, key=lambda r: r["objective"])

        return {
            "status": "success",
            "message": f"Vanilla Bayesian Optimization with BoTorch finished using {self.aqf}.",
            "best_x": best_record["x"],
            "best_objective": float(best_record["objective"]),
            "outputs": best_record["outputs"],
            "evaluations": self.history_records
        }, self.history_records

