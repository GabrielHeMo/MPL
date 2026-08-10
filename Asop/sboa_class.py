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

RUN_STATUS_DIR = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
FAILED_RUN_STATUS_VALUES = {9, 10}
PENALTY_VALUE = 1e10



class SBOA():
    def __init__(self, config, aspen):
        self.config = config
        self.aspen = aspen
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

    def sboa_function(self, fobj, lb, ub, dim, pop_size=50, max_iter=1000,
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

    def run(self):
        history_records = []

        lb = np.array([float(v["lower_bound"]) for v in self.config["inputs"]])
        ub = np.array([float(v["upper_bound"]) for v in self.config["inputs"]])
        dim = len(self.config["inputs"])

        hyper = self.config["hyperparameters"]

        def fobj(x):
            result = self.evaluate_point(self.aspen, self.config, x)

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
            return objective_value

        best_pos, best_score, convergence = self.sboa_function(
                                                        fobj=fobj,
                                                        lb=lb,
                                                        ub=ub,
                                                        dim=dim,
                                                        pop_size=int(hyper.get("pop_size", 30)),
                                                        max_iter=int(hyper.get("max_iter", hyper.get("generations", 50))),
                                                        Cp1=float(hyper.get("Cp1", 0.9)),
                                                        Cp2=float(hyper.get("Cp2", 0.55)),
                                                        seed=int(hyper.get("seed", 1))
                                                    )

        valid_records = [
            row for row in history_records
            if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
        ]

        if len(valid_records) == 0:
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
