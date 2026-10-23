# -*- coding: utf-8 -*-
"""
Success-Based Optimization Algorithm (SBOA)

Implementacion de referencia autocontenida del algoritmo propuesto en:

    Lara-Montano, O.D., Gomez-Castro, F.I., Gutierrez-Antonio, C.,
    Dragoi, E.N. (2025). Success-Based Optimization Algorithm (SBOA):
    Development and enhancement of a metaheuristic optimizer.
    Computers and Chemical Engineering, 194, 108987.
    https://doi.org/10.1016/j.compchemeng.2024.108987

Inspirado en la teoria de atribucion del exito. La busqueda se guia por dos
poblaciones de influencia:
    spA : las tres mejores soluciones actuales (explotacion).
    spB : una subpoblacion producida por torneo binario (exploracion).
Un criterio probabilistico (Cp1, Cp2) decide que poblacion influye en cada
solucion candidata en cada iteracion.

Coeficientes de transicion exploracion -> explotacion (Eqs. 9-10):
    D1(t) = 1 - t^2 / N^2
    D2(t) = 1 - t / N
"""

from __future__ import annotations
import json
from pickle import FALSE, FLOAT
from pyclbr import Class
import sys
import time
import traceback
import numpy as np
import time
import math

from .status import emit_status
from Evaluation.remote_aspen_problem_functions import CSharpAspenEvaluator

RUN_STATUS_DIR = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
FAILED_RUN_STATUS_VALUES = {9, 10}
PENALTY_VALUE = 1e10



class SBOA():
    def __init__(self, config):
        self.config = config
        self.evaluator = CSharpAspenEvaluator(config)
        self.history_records = []

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


    def evaluate_point(self, config, x):
        """Evaluate one SBOA candidate through the C# -> Aspen JSONL bridge."""
        x = self.preprocess_x(config, x)
        try:
            return self.evaluator.evaluate_one(x)
        except Exception as e:
            return self.penalty_result(
                config=config,
                x=x,
                reason=f"Error requesting Aspen evaluation from C#: {e}",
            )
    def sboa_function(self, fobj, lb, ub, dim, pop_size=50, max_iter=1000,
         Cp1=0.9, Cp2=0.55, seed=None, report_status=False):
        """Minimiza la funcion objetivo fobj mediante SBOA.

        Parametros
        ----------
        fobj : callable
            Funcion objetivo a minimizar. Recibe un vector de longitud dim y
            retorna un escalar.
        lb, ub : float | array_like
            Limites inferior y superior del dominio. Escalares (mismo limite en
            todas las dimensiones) o arreglos de longitud dim.
        dim : int
            Numero de variables de decision.
        pop_size : int, opcional
            Numero de soluciones candidatas (n). Por defecto 50.
        max_iter : int, opcional
            Numero de iteraciones (N). Por defecto 1000.
        Cp1, Cp2 : float, opcional
            Criterios probabilisticos. Valores optimos del articulo (Tabla 3):
            Cp1 = 0.9, Cp2 = 0.55.
        seed : int | None, opcional
            Semilla para reproducibilidad.

        Retorna
        -------
        best_pos : np.ndarray
            Mejor vector de solucion encontrado.
        best_score : float
            Valor de la funcion objetivo en best_pos.
        history : np.ndarray
            Mejor valor por iteracion (curva de convergencia), longitud max_iter.
        """
        rng = np.random.default_rng(seed)

        lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()

        # Inicializacion (Eq. 2)
        if report_status:
            emit_status(
                f"SBOA initial population: evaluating {pop_size} candidates.",
                level="INFO",
                progress=0,
                stage="initial_design",
                current=0,
                total=pop_size,
                algorithm="SBOA",
            )

        X = lb + rng.random((pop_size, dim)) * (ub - lb)
        fitness = np.array([fobj(X[i]) for i in range(pop_size)])

        best_idx = int(np.argmin(fitness))
        best_pos = X[best_idx].copy()
        best_score = float(fitness[best_idx])

        if report_status:
            emit_status(
                f"SBOA initial population completed. Best objective = {best_score:.6g}.",
                level="PROGRESS",
                progress=100.0 / max(1, max_iter + 1),
                stage="initial_design",
                current=pop_size,
                total=pop_size,
                algorithm="SBOA",
            )

        history = np.empty(max_iter)

        for t in range(max_iter):
            # Coeficientes de transicion (Eqs. 9-10)
            D1 = 1.0 - (t ** 2) / (max_iter ** 2)
            D2 = 1.0 - t / max_iter

            # spA: las tres mejores soluciones (Eq. 4)
            orden = np.argsort(fitness)
            spA = X[orden[:3]].copy()

            # spB: subpoblacion por torneo binario (Fig. 1)
            a = rng.integers(pop_size, size=pop_size)
            b = rng.integers(pop_size, size=pop_size)
            ganadores = np.where(fitness[a] <= fitness[b], a, b)
            spB = X[ganadores].copy()
            spB_fit = fitness[ganadores]

            # SPB_avg: miembro de spB mas cercano al exito promedio de spB
            SPB_avg = spB[int(np.argmin(np.abs(spB_fit - spB_fit.mean())))]

            for i in range(pop_size):
                if rng.random() <= Cp1:
                    # Influencia de spA: candidata greedy G (Eq. 5)
                    r2 = rng.random(dim)
                    r3 = rng.random(dim)
                    rand = spA[rng.integers(3)]
                    G = spA[0] + D1 * (2 * r2 - 1) * (2 * r3) * np.abs(rand - X[i])
                    G = np.clip(G, lb, ub)
                    G_fit = fobj(G)
                    if G_fit <= fitness[i]:          # aceptacion greedy
                        X[i] = G
                        fitness[i] = G_fit
                else:
                    # Influencia de spB
                    r4 = rng.random(dim)
                    r5 = rng.random(dim)
                    if rng.random() <= Cp2:
                        referencia = spB[rng.integers(pop_size)]   # Eq. 7
                    else:
                        referencia = SPB_avg                       # Eq. 8
                    X_new = X[i] + D2 * (2 * r4 - 1) * np.abs((2 * r5) * referencia - X[i])
                    X_new = np.clip(X_new, lb, ub)
                    X[i] = X_new
                    fitness[i] = fobj(X_new)

                if fitness[i] < best_score:
                    best_score = float(fitness[i])
                    best_pos = X[i].copy()

            history[t] = best_score

            if report_status:
                emit_status(
                    f"SBOA iteration {t + 1}/{max_iter} completed. Best objective = {best_score:.6g}.",
                    level="PROGRESS",
                    progress=100.0 * (t + 2) / max(1, max_iter + 1),
                    stage="optimization",
                    current=t + 1,
                    total=max_iter,
                    algorithm="SBOA",
                )

        return best_pos, best_score, history

    def run(self):
        history_records = []

        lb = np.array([float(v["lower_bound"]) for v in self.config["inputs"]])
        ub = np.array([float(v["upper_bound"]) for v in self.config["inputs"]])
        dim = len(self.config["inputs"])

        hyper = self.config["hyperparameters"]

        pop_size = int(hyper.get("pop_size", 30))
        max_iter = int(hyper.get("max_iter", hyper.get("generations", 50)))
        cp1 = float(hyper.get("Cp1", 0.9))
        cp2 = float(hyper.get("Cp2", 0.55))
        seed = int(hyper.get("seed", 1))

        emit_status(
            f"Starting SBOA: pop_size={pop_size}, iterations={max_iter}, Cp1={cp1}, Cp2={cp2}, seed={seed}.",
            level="INFO",
            progress=0,
            stage="setup",
            algorithm="SBOA",
        )

        def fobj(x):
            result = self.evaluate_point(self.config, x)

            x_used = result["x"]
            objective_value = float(result["objective"])
            outputs = result["outputs"]

            record = {
                "evaluation": len(history_records) + 1,
                "x": {
                    self.config["inputs"][i]["tag"]: float(x_used[i])
                    for i in range(len(x_used))
                },
                "objective": objective_value,
                "outputs": outputs,
                "feasible": bool(result["feasible"]),
                "aspen_converged": bool(result["aspen_converged"]),
                "run_status": result["run_status"],
                "message": result["message"],
                #"violations": result["violations"]
            }

            history_records.append(record)

            emit_status(
                f"SBOA Aspen evaluation {record['evaluation']} completed: objective={objective_value:.6g}, "
                f"converged={record['aspen_converged']}, feasible={record['feasible']}.",
                level="PROGRESS" if record["aspen_converged"] else "WARNING",
                stage="aspen_evaluation",
                current=record["evaluation"],
                log=False,
                algorithm="SBOA",
            )
            return objective_value

        best_pos, best_score, convergence = self.sboa_function(
                                                        fobj=fobj,
                                                        lb=lb,
                                                        ub=ub,
                                                        dim=dim,
                                                        pop_size=pop_size,
                                                        max_iter=max_iter,
                                                        Cp1=cp1,
                                                        Cp2=cp2,
                                                        seed=seed,
                                                        report_status=True
                                                    )

        valid_records = [
            row for row in history_records
            if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
        ]

        if len(valid_records) == 0:
            emit_status(
                "SBOA finished without any feasible Aspen-converged point.",
                level="ERROR",
                progress=100,
                stage="finished",
                algorithm="SBOA",
            )
            return {
                "status": "error",
                "message": "SBOA terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "best_x": None,
                "best_objective": None,
                "outputs": None,
                "evaluations": history_records,
                "convergence": convergence.tolist()
            }, history_records

        best_record = min(valid_records, key=lambda r: r["objective"])

        emit_status(
            f"SBOA finished successfully. Best objective = {float(best_record['objective']):.6g}.",
            level="SUCCESS",
            progress=100,
            stage="finished",
            algorithm="SBOA",
        )

        return {
            "status": "success",
            "message": "SBOA optimization finished.",
            "best_x": best_record["x"],
            "best_objective": float(best_record["objective"]),
            "outputs": best_record["outputs"],
            "evaluations": history_records,
            "convergence": convergence.tolist()
        }, history_records



def sboa(fobj, lb, ub, dim, pop_size=50, max_iter=1000,
         Cp1=0.9, Cp2=0.55, seed=None):
    """Minimiza la funcion objetivo fobj mediante SBOA.

    Parametros
    ----------
    fobj : callable
        Funcion objetivo a minimizar. Recibe un vector de longitud dim y
        retorna un escalar.
    lb, ub : float | array_like
        Limites inferior y superior del dominio. Escalares (mismo limite en
        todas las dimensiones) o arreglos de longitud dim.
    dim : int
        Numero de variables de decision.
    pop_size : int, opcional
        Numero de soluciones candidatas (n). Por defecto 50.
    max_iter : int, opcional
        Numero de iteraciones (N). Por defecto 1000.
    Cp1, Cp2 : float, opcional
        Criterios probabilisticos. Valores optimos del articulo (Tabla 3):
        Cp1 = 0.9, Cp2 = 0.55.
    seed : int | None, opcional
        Semilla para reproducibilidad.

    Retorna
    -------
    best_pos : np.ndarray
        Mejor vector de solucion encontrado.
    best_score : float
        Valor de la funcion objetivo en best_pos.
    history : np.ndarray
        Mejor valor por iteracion (curva de convergencia), longitud max_iter.
    """
    rng = np.random.default_rng(seed)

    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()

    # Inicializacion (Eq. 2)
    X = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([fobj(X[i]) for i in range(pop_size)])

    best_idx = int(np.argmin(fitness))
    best_pos = X[best_idx].copy()
    best_score = float(fitness[best_idx])

    history = np.empty(max_iter)

    for t in range(max_iter):
        # Coeficientes de transicion (Eqs. 9-10)
        D1 = 1.0 - (t ** 2) / (max_iter ** 2)
        D2 = 1.0 - t / max_iter

        # spA: las tres mejores soluciones (Eq. 4)
        orden = np.argsort(fitness)
        spA = X[orden[:3]].copy()

        # spB: subpoblacion por torneo binario (Fig. 1)
        a = rng.integers(pop_size, size=pop_size)
        b = rng.integers(pop_size, size=pop_size)
        ganadores = np.where(fitness[a] <= fitness[b], a, b)
        spB = X[ganadores].copy()
        spB_fit = fitness[ganadores]

        # SPB_avg: miembro de spB mas cercano al exito promedio de spB
        SPB_avg = spB[int(np.argmin(np.abs(spB_fit - spB_fit.mean())))]

        for i in range(pop_size):
            if rng.random() <= Cp1:
                # Influencia de spA: candidata greedy G (Eq. 5)
                r2 = rng.random(dim)
                r3 = rng.random(dim)
                rand = spA[rng.integers(3)]
                G = spA[0] + D1 * (2 * r2 - 1) * (2 * r3) * np.abs(rand - X[i])
                G = np.clip(G, lb, ub)
                G_fit = fobj(G)
                if G_fit <= fitness[i]:          # aceptacion greedy
                    X[i] = G
                    fitness[i] = G_fit
            else:
                # Influencia de spB
                r4 = rng.random(dim)
                r5 = rng.random(dim)
                if rng.random() <= Cp2:
                    referencia = spB[rng.integers(pop_size)]   # Eq. 7
                else:
                    referencia = SPB_avg                       # Eq. 8
                X_new = X[i] + D2 * (2 * r4 - 1) * np.abs((2 * r5) * referencia - X[i])
                X_new = np.clip(X_new, lb, ub)
                X[i] = X_new
                fitness[i] = fobj(X_new)

            if fitness[i] < best_score:
                best_score = float(fitness[i])
                best_pos = X[i].copy()

        history[t] = best_score

    return best_pos, best_score, history


if __name__ == "__main__":
    # Demostracion: minimizacion de funciones de prueba estandar.
    def sphere(x):
        return float(np.sum(x ** 2))

    def rastrigin(x):
        x = np.asarray(x)
        return float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    dim = 30
    for nombre, f, lim in [("Sphere", sphere, 5.12),
                           ("Rastrigin", rastrigin, 5.12)]:
        pos, score, hist = sboa(f, -lim, lim, dim,
                                pop_size=50, max_iter=1000, seed=42)
        print(f"{nombre:10s} | mejor f = {score:.6e} | "
              f"||x|| = {np.linalg.norm(pos):.6e}")
