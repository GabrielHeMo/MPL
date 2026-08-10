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

import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.spx import SPX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultSingleObjectiveTermination
# from pymoo.util import default_random_state


PENALTY_VALUE = 1e10
RUN_STATUS_DIR = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
FAILED_RUN_STATUS_VALUES = {9, 10}


class AspenProblemMOBO(ElementwiseProblem):

    def __init__(self, config, aspen):
        self.config = config
        self.aspen = aspen
        self.history = []

        self.objectives = config.get("objectives", []) or []

        if len(self.objectives) < 2:
            raise ValueError(
                "NSGA-II requiere al menos 2 funciones objetivo en config['objectives']."
            )

        self.objective_names = [
            self.get_objective_name(obj, i)
            for i, obj in enumerate(self.objectives)
        ]

        xl = np.array([float(v["lower_bound"]) for v in config["inputs"]])
        xu = np.array([float(v["upper_bound"]) for v in config["inputs"]])

        super().__init__(
            n_var=len(config["inputs"]),
            n_obj=len(self.objectives),
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )
    
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

    def penalty_result(self, config, x, reason, violations=None, run_status=None):
        x = self.preprocess_x(config, x)
        outputs = {output_var["tag"]: None for output_var in config.get("outputs", [])}

        objective_values = [float(PENALTY_VALUE)] * len(self.objectives)
        #objectives_dict = {
        #    name: float(PENALTY_VALUE)
        #    for name in self.objective_names
        #}

        #objectives_raw_dict = {
        #    name: None
        #    for name in self.objective_names
        #}

        return {
            "objective": float(PENALTY_VALUE),  # compatibilidad con código viejo
            "objective_values": objective_values,
            "outputs": outputs,
            "x": x,
            "feasible": False,
            "aspen_converged": False,
            "run_status": run_status,
            "message": reason,
            #"violations": violations or []
        }

    def set_aspen_variable(self,aspen, path, value):
        node = aspen.Tree.FindNode(path)

        if node is None:
            raise RuntimeError(f"The variable Aspen was not found.: {path}")

        node.Value = value


    def get_aspen_variable(self,aspen, path):
        node = aspen.Tree.FindNode(path)
        if node is None:
            raise RuntimeError(f"Aspen exit not found: {path}")
        if node.Value is None:
            raise RuntimeError(f"The Aspen output has no value: {path}")
        value = float(node.Value)
        if not np.isfinite(value):
            raise RuntimeError(f"The Aspen output is not finite: {path} = {value}")
        return value

    
    def update_aspen(self,aspen, config):
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

    
    def input_context(self,config, x):
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

    def resolve_value(self,token, ctx):
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
    
    def compare_values(self, left, operator, right, tol=1e-9):
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
     
    def check_input_constraints(self,config, x):
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

    def run_aspen(self,aspen):
        aspen.Run2()

        try:
            while aspen.Engine.IsRunning:
                time.sleep(0.2)
        except Exception:
            time.sleep(2)

    def reinit_aspen(self,aspen):
        try:
            aspen.Reinit()
        except Exception:
            pass


    def check_aspen_convergence(self,aspen):
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

       
    def evaluate_point(self,aspen, config, x):
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

            (
                objective_values,
                objective_values_raw,
                objectives_dict,
                objectives_raw_dict
            ) = self.evaluate_objectives(config, x, outputs)

            return {
                "objective": float(objective_values[0]),  # compatibilidad con código viejo
                "objective_values": objective_values,
                "outputs": outputs,
                "x": x,
                "feasible": True,
                "aspen_converged": True,
                "run_status": run_status,
                "message": status_message,}

        except Exception as e:
            self.reinit_aspen(aspen)
            return self.penalty_result(
                config=config,
                x=x,
                reason=f"Error al evaluar el punto en Aspen: {e}"
            )

    def get_objective_name(self, obj, i):
        """
        Nombre amigable para guardar cada objetivo en history/results.
        """
        name = (
            obj.get("tag")
            or obj.get("alias")
            or obj.get("name")
            or obj.get("objective_name")
            or f"f{i + 1}"
        )
        return str(name)

    def get_objective_sense(self, obj):
        """
        pymoo minimiza por defecto.
        Si el usuario pide maximizar, se multiplica por -1.
        """
        sense = (
            obj.get("sense")
            or obj.get("Sense")
            or obj.get("type")
            or obj.get("Type")
            or "minimize"
        )

        sense = str(sense).strip().lower()

        if sense in ["minimize", "min", "minimum", "minimizar"]:
            return "minimize"

        if sense in ["maximize", "max", "maximum", "maximizar"]:
            return "maximize"

        raise ValueError(f"Sentido de objetivo no soportado: {sense}")

    def evaluate_objectives(self, config, x, outputs):
        """
        Evalúa todas las funciones objetivo de config['objectives'].

        Regresa:
        - objective_values: valores transformados para pymoo, todos en modo minimización.
        - objective_values_raw: valores reales, sin cambiar signo.
        - objectives_dict: dict con valores usados por pymoo.
        - objectives_raw_dict: dict con valores reales.
        """
        input_ctx = self.input_context(config, x)

        safe_globals = {"__builtins__": {}}
        safe_locals = {
            **input_ctx,
            **outputs,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "math": math,
            "np": np,
        }

        objective_values = []
        objective_values_raw = []

        for i, obj in enumerate(self.objectives):
            expression = obj.get("expression") or obj.get("Expression")

            if expression is None or str(expression).strip() == "":
                raise RuntimeError(f"El objetivo {i + 1} no tiene expresión.")

            raw_value = float(eval(expression, safe_globals, safe_locals))

            if not np.isfinite(raw_value):
                raise RuntimeError(
                    f"La función objetivo {self.objective_names[i]} no es finita: {raw_value}"
                )

            sense = self.get_objective_sense(obj)

            if sense == "maximize":
                pymoo_value = -raw_value
            else:
                pymoo_value = raw_value

            objective_values_raw.append(raw_value)
            objective_values.append(pymoo_value)

        objectives_dict = {
            name: float(value)
            for name, value in zip(self.objective_names, objective_values)
        }

        objectives_raw_dict = {
            name: float(value)
            for name, value in zip(self.objective_names, objective_values_raw)
        }

        return objective_values, objective_values_raw, objectives_dict, objectives_raw_dict

    def restore_raw_objectives(self, F):
        """
        Convierte res.F de pymoo a valores reales.
        Útil porque los objetivos Max fueron multiplicados por -1.
        """
        if F is None:
            return None

        F = np.asarray(F, dtype=float)

        if F.ndim == 1:
            F = F.reshape(1, -1)

        raw = F.copy()

        for j, obj in enumerate(self.objectives):
            if self.get_objective_sense(obj) == "maximize":
                raw[:, j] = -raw[:, j]

        return raw

    def _evaluate(self, x, out, *args, **kwargs):
        result = self.evaluate_point(self.aspen, self.config, x)

        x_used = result["x"]
        objective_values = np.array(result["objective_values"], dtype=float)
        outputs = result["outputs"]


        record = {  "evaluation": len(self.history) + 1,
                    "x": {
                        self.config["inputs"][i]["tag"]: float(x_used[i])
                        for i in range(len(x_used))
                    },
                    "objective": float(objective_values[0]),
                    "objective_values": objective_values.tolist(),
                    "outputs": outputs,
                    "feasible": bool(result.get("feasible", False)),
                    "aspen_converged": bool(result.get("aspen_converged", False)),
                    "run_status": result.get("run_status"),
                    "message": result.get("message", "")
                }
        self.history.append(record)

        # Esto es lo más importante para NSGA-II:
        out["F"] = objective_values

class AspenProblem(ElementwiseProblem):

    def __init__(self, config, aspen):
        self.config = config
        self.aspen = aspen
        self.history = []

        xl = np.array([float(v["lower_bound"]) for v in config["inputs"]])
        xu = np.array([float(v["upper_bound"]) for v in config["inputs"]])

        super().__init__(
            n_var=len(config["inputs"]),
            n_obj=1,
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )
    
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

    def penalty_result(self,config, x, reason, violations=None, run_status=None):
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

    def set_aspen_variable(self,aspen, path, value):
        node = aspen.Tree.FindNode(path)

        if node is None:
            raise RuntimeError(f"The variable Aspen was not found.: {path}")

        node.Value = value

    def get_aspen_variable(self,aspen, path):
        node = aspen.Tree.FindNode(path)
        if node is None:
            raise RuntimeError(f"Aspen exit not found: {path}")
        if node.Value is None:
            raise RuntimeError(f"The Aspen output has no value: {path}")
        value = float(node.Value)
        if not np.isfinite(value):
            raise RuntimeError(f"The Aspen output is not finite: {path} = {value}")
        return value
 
    def update_aspen(self,aspen, config):
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
    
    def input_context(self,config, x):
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

    def resolve_value(self,token, ctx):
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
    
    def compare_values(self, left, operator, right, tol=1e-9):
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
     
    def check_input_constraints(self,config, x):
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

    def run_aspen(self,aspen):
        aspen.Run2()

        try:
            while aspen.Engine.IsRunning:
                time.sleep(0.2)
        except Exception:
            time.sleep(2)

    def reinit_aspen(self,aspen):
        try:
            aspen.Reinit()
        except Exception:
            pass

    def check_aspen_convergence(self,aspen):
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
     
    def evaluate_point(self,aspen, config, x):
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

    def _evaluate(self, x, out, *args, **kwargs):
        result = self.evaluate_point(self.aspen, self.config, x)

        x_used = result["x"]
        objective_value = float(result["objective"])
        outputs = result["outputs"]

 

        record = {
                        "evaluation": len(self.history) + 1,
                        "x": {
                            self.config["inputs"][i]["tag"]: float(x_used[i])
                            for i in range(len(x_used))
                        },
                        "objective": objective_value,
                        "outputs": outputs,
                        "feasible": bool(result.get("feasible", False)),
                        "aspen_converged": bool(result.get("aspen_converged", False)),
                        "run_status": result.get("run_status"),
                        "message": result.get("message", "")
                    }
              
        self.history.append(record)

        out["F"] = objective_value
