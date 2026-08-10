import json
from pickle import FALSE, FLOAT
from pyclbr import Class
import sys
import time
import traceback
import re
from jaxtyping import Float
import numpy as np
import win32com.client
import csv
import os 
import math
import torch
from pymoo.core.problem import ElementwiseProblem
# ALGORITMOS MONO-OBJETIVO 
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead, initialize_simplex
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES

from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.nrbo import NRBO

# ALGORITMOS MULTI-OBJETIVO 
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
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
from pymoo.util.ref_dirs import get_reference_directions
from cmaes_setup import CMAES , initialize_CMAES # type: ignore

from pymoo.optimize import minimize

## UTILITIES
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

# Export other clases  
from sboa_class import SBOA # type: ignore
from aspenproblems import AspenProblem , AspenProblemMOBO # type: ignore
from bayesian_classes import DiscreteBOBotorch, VanillaBOBotorch # type: ignore
from TURBO_class import TrustRegionBOBotorch , ContinuousMOBOBotorch# type: ignore
from MORPH_class import MorphBOBotorch, MorphMOBOBotorch #type: ignore

RUN_STATUS_DIR = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
FAILED_RUN_STATUS_VALUES = {9, 10}
PENALTY_VALUE = 1e10

def open_aspen_file(file_path):
    try:
        aspen = win32com.client.Dispatch("Apwn.Document")
        aspen.InitFromArchive2(file_path)
        aspen.Visible = False
        try:
            aspen.SuppressDialogs = True
        except Exception:
            pass
        return aspen
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir Aspen desde Python: {e}")


def close_aspen(aspen):
    if aspen is None:
        return

    try:
        aspen.Close(False)
    except Exception:
        pass

    try:
        del aspen
    except Exception:
        pass



def write_history_csv_mono(config, results_path, history):
    import os
    import csv

    csv_path = os.path.splitext(results_path)[0] + "_evaluations.csv"

    input_tags = [
        v.get("tag") or v.get("alias") or f"x{i + 1}"
        for i, v in enumerate(config.get("inputs", []))
    ]

    output_tags = [
        v.get("tag") or v.get("alias") or f"y{i + 1}"
        for i, v in enumerate(config.get("outputs", []))
    ]

    objectives_config = config.get("objectives", []) or []

    if len(objectives_config) > 0:
        obj = objectives_config[0]
        objective_name = (
            obj.get("tag")
            or obj.get("alias")
            or obj.get("name")
            or obj.get("objective_name")
            or "obj1"
        )
    else:
        objective_name = "obj1"

    extra_keys = [
        "algorithm",
        "acquisition_function_raw",
        "acquisition_function",
        "turbo_iteration",
        "candidate_source",
        "trust_region_length",
        "acq_value",
        "tr_lb",
        "tr_ub",
        "feasible",
        "aspen_converged",
        "run_status",
    ]

    header = (
        ["evaluation"]
        + [f"variables.{tag}" for tag in input_tags]
        + [f"outputs.{tag}" for tag in output_tags]
        + [f"objectives.{objective_name}"]
        + extra_keys
        + ["message"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in history:
            x_dict = row.get("x", {}) or {}
            outputs_dict = row.get("outputs", {}) or {}

            variable_values = [
                x_dict.get(tag)
                for tag in input_tags
            ]

            output_values = [
                outputs_dict.get(tag)
                for tag in output_tags
            ]

            extra_values = [
                row.get(key)
                for key in extra_keys
            ]

            writer.writerow(
                [row.get("evaluation")]
                + variable_values
                + output_values
                + [row.get("objective")]
                + extra_values
                + [row.get("message")]
            )

    return csv_path


def write_history_csv_multi(config, results_path, history):
    import os
    import csv

    csv_path = os.path.splitext(results_path)[0] + "_evaluations.csv"

    input_tags = [
        v.get("tag") or v.get("alias") or f"x{i + 1}"
        for i, v in enumerate(config.get("inputs", []))
    ]

    output_tags = [
        v.get("tag") or v.get("alias") or f"y{i + 1}"
        for i, v in enumerate(config.get("outputs", []))
    ]

    objectives_config = config.get("objectives", []) or []

    objective_names = []

    for i, obj in enumerate(objectives_config):
        name = (
            obj.get("tag")
            or obj.get("alias")
            or obj.get("name")
            or obj.get("objective_name")
            or f"obj{i + 1}"
        )
        objective_names.append(str(name))

    if len(objective_names) == 0:
        objective_names = ["obj1"]

    seen = {}

    for i, name in enumerate(objective_names):
        if name not in seen:
            seen[name] = 1
        else:
            seen[name] += 1
            objective_names[i] = f"{name}_{seen[name]}"

    header = (
                ["evaluation"]
                + [f"variables.{tag}" for tag in input_tags]
                + [f"outputs.{tag}" for tag in output_tags]
                + [f"objectives.{name}" for name in objective_names]
                + ["feasible", "message"]
                )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in history:
            x_dict = row.get("x", {}) or {}
            outputs_dict = row.get("outputs", {}) or {}

            variable_values = [
                x_dict.get(tag)
                for tag in input_tags
            ]

            output_values = [
                outputs_dict.get(tag)
                for tag in output_tags
            ]

            objective_values = row.get("objective_values", [])

            if objective_values is None:
                objective_values = []

            objective_values = list(objective_values)

            while len(objective_values) < len(objective_names):
                objective_values.append(None)

            objective_values = objective_values[:len(objective_names)]

            writer.writerow(
                [row.get("evaluation")]
                + variable_values
                + output_values
                + objective_values
                + [
                    row.get("feasible"),
                    row.get("message"),
                ]
            )

    return csv_path

# Auviliary functions

def default_n_ref_dirs(n_obj):
    """
    Número razonable de direcciones de referencia para problemas caros tipo Aspen.
    No uses valores enormes porque cada generación evalúa muchas simulaciones.
    """
    if n_obj <= 2:
        return 80
    elif n_obj == 3:
        return 92
    elif n_obj == 4:
        return 120
    elif n_obj == 5:
        return 160
    else:
        return 200


def build_reference_directions_for_nsga3(config, problem, seed):
    """
    Crea las reference directions para NSGA-III.

    Importante:
    - ref_dirs depende del número de objetivos.
    - No depende directamente de la magnitud de TAC, CO2, FEDI, IRR, etc.
    """
    hyper = config["hyperparameters"]

    # Más seguro usar problem.n_obj, porque ya viene de AspenProblemMOBO.
    n_obj = int(getattr(problem, "n_obj", len(config["objectives"])))

    ref_dir_method = str(hyper.get("ref_dir_method", "energy")).lower()

    if ref_dir_method == "energy":
        n_ref_dirs = int(
            hyper.get(
                "n_ref_dirs",
                hyper.get("pop_size", default_n_ref_dirs(n_obj))
            )
        )

        ref_dirs = get_reference_directions("energy",
                                              n_obj,
                                              n_ref_dirs,
                                              seed=seed)
        return  ref_dirs
                                                 

    elif ref_dir_method in ["das-dennis", "das_dennis", "uniform"]:
        n_partitions = int(hyper.get("n_partitions", 12))

        ref_dirs = get_reference_directions( "das-dennis",
                                                n_obj,
                                                n_partitions=n_partitions
                                            )
        return  ref_dirs

    else:
        raise ValueError(f"Unknown ref_dir_method for NSGA-III: {ref_dir_method}")


def str_to_bool(value, default=False):
    """
    Convierte valores enviados desde C# o JSON a bool.
    Acepta True/False, true/false, 1/0, yes/no, si/no.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ["true", "1", "yes", "y", "si", "sí"]:
        return True
    if text in ["false", "0", "no", "n"]:
        return False
    return default


def parse_numeric_vector_or_matrix(raw, name="value"):
    """
    Lee un vector o matriz numérica desde JSON/lista/string.

    Formatos aceptados:
    - [[0.5, 0.2], [0.1, 0.6]]
    - "[[0.5, 0.2], [0.1, 0.6]]"
    - "0.5,0.2;0.1,0.6"
    - "0.5 0.2; 0.1 0.6"
    """
    if raw is None:
        return None

    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None

        # Primero intenta leerlo como JSON real.
        try:
            return np.array(json.loads(text), dtype=float)
        except Exception:
            pass

        rows = []
        for row_text in text.replace("|", ";").split(";"):
            row_text = row_text.strip()
            if row_text == "":
                continue
            values = [v for v in re.split(r"[\s,]+", row_text) if v.strip() != ""]
            rows.append([float(v) for v in values])

        if len(rows) == 0:
            raise ValueError(f"No se pudo interpretar {name}: {raw}")

        return np.array(rows, dtype=float)

    return np.array(raw, dtype=float)


def build_reference_points_for_rnsga(config, problem):
    """
    Construye los reference/aspiration points para R-NSGA-II y R-NSGA-III.

    IMPORTANTE:
    Estos puntos deben estar en el mismo espacio que usa pymoo para minimizar.
    Si en tu GUI declaras un objetivo como Maximize, AspenProblemMOBO normalmente
    lo transforma internamente a minimización; por lo tanto, el punto de referencia
    debe respetar esa convención interna.
    """
    hyper = config.get("hyperparameters", {}) or {}
    n_obj = int(getattr(problem, "n_obj", len(config.get("objectives", []))))

    raw = (
        hyper.get("ref_points", None)
        or hyper.get("reference_points", None)
        or hyper.get("aspiration_points", None)
    )

    ref_points = parse_numeric_vector_or_matrix(raw, name="ref_points")

    if ref_points is None:
        raise ValueError(
            "R-NSGA-II/R-NSGA-III requieren ref_points. "
            "Ejemplo para 2 objetivos: [[0.5, 0.2], [0.1, 0.6]] "
            "o como texto: 0.5,0.2;0.1,0.6. "
            "Cada fila debe tener el mismo número de columnas que objetivos."
        )

    if ref_points.ndim == 1:
        ref_points = ref_points.reshape(1, -1)

    if ref_points.ndim != 2:
        raise ValueError("ref_points debe ser una matriz 2D: filas=puntos, columnas=objetivos.")

    if ref_points.shape[1] != n_obj:
        raise ValueError(
            f"ref_points tiene {ref_points.shape[1]} columnas, "
            f"pero el problema tiene {n_obj} objetivos."
        )

    return ref_points.astype(float)


def build_weights_for_rnsga(config, problem):
    """
    Pesos opcionales para R-NSGA-II.
    Si no se mandan desde C#, se usa None y pymoo aplica su valor por defecto.
    """
    hyper = config.get("hyperparameters", {}) or {}
    n_obj = int(getattr(problem, "n_obj", len(config.get("objectives", []))))

    weights = parse_numeric_vector_or_matrix(hyper.get("weights", None), name="weights")

    if weights is None:
        return None

    weights = np.array(weights, dtype=float).reshape(-1)

    if len(weights) != n_obj:
        raise ValueError(
            f"weights tiene {len(weights)} valores, pero el problema tiene {n_obj} objetivos."
        )

    return weights


def get_variable_type_info(config):
    list_variable_types = []
    for input_var in config.get("inputs", []):
        variable_type = input_var.get("variable_type", "Continuous")
        list_variable_types.append(str(variable_type))

    unique_list = list(set(list_variable_types))
    unique_list_lower = [v.lower() for v in unique_list]
    only_continuous = len(unique_list_lower) == 1 and unique_list_lower[0] == "continuous"
    only_discrete = len(unique_list_lower) == 1 and unique_list_lower[0] == "discrete"
    mixed = len(unique_list_lower) > 1

    return unique_list, unique_list_lower, only_continuous, only_discrete, mixed


def make_multiobjective_results(config, problem, res, extra=None):
    pareto_F_raw = problem.restore_raw_objectives(res.F) if res.F is not None else None

    results = {
        "algorithm": config["algorithm"],
        "problem_type": config["problem_type"],
        "objective_names": problem.objective_names,
        "pareto_F_minimized": res.F.tolist() if res.F is not None else [],
        "pareto_F_raw": pareto_F_raw.tolist() if pareto_F_raw is not None else [],
        "pareto_X": res.X.tolist() if res.X is not None else [],
        "history": problem.history
    }

    if extra:
        results.update(extra)

    return results


def hyper_value(hyper, key, default=None):
    """Lee un hiperparámetro enviado desde C# y permite usar defaults si viene vacío."""
    value = hyper.get(key, None)
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return value


def hyper_int(hyper, key, default):
    value = hyper_value(hyper, key, default)
    return int(float(value))


def hyper_float(hyper, key, default):
    value = hyper_value(hyper, key, default)
    return float(value)


def hyper_text(hyper, key, default):
    value = hyper_value(hyper, key, default)
    return str(value)


def warn_if_not_continuous_for_moo(algorithm_name, only_continuous):
    if not only_continuous:
        print(
            f"WARNING: {algorithm_name} trabaja mejor con variables continuas. "
            "Si usas variables discretas, AspenProblemMOBO debe redondearlas/repararlas internamente.",
            flush=True
        )


def save_multiobjective_run(config, results_path, problem, res, extra=None):
    results = make_multiobjective_results(config, problem, res, extra=extra)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    write_history_csv_multi(config, results_path, problem.history)




def main():

    config_path = sys.argv[1]
    results_path = sys.argv[2]
    aspen = None
    # REVISAR CON QUE ALGORITMO CORRE
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        algorithm_name = str(config.get("algorithm", "")).strip()

        problem_type_raw = str(config.get("problem_type", "")).strip()

        problem_type = (
            problem_type_raw
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        #print("========== DEBUG PYTHON ==========", flush=True)
        #print(f"algorithm raw     = {config.get('algorithm')}", flush=True)
        #print(f"algorithm_name    = {algorithm_name}", flush=True)
        #print(f"problem_type raw  = {problem_type_raw}", flush=True)
        #print(f"problem_type norm = {problem_type}", flush=True)
        #print(f"n_objectives      = {len(config.get('objectives', []))}", flush=True)
        #print("objectives:", flush=True)
        #print(json.dumps(config.get("objectives", []), indent=4), flush=True)
        #print("==================================", flush=True)
        if algorithm_name == 'PSO' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)

            hyper = config["hyperparameters"]
            algorithm = PSO(
                pop_size=int(hyper["pop_size"]),
                w=float(hyper["w"]),
                c1=float(hyper["c1"]),
                c2=float(hyper["c2"]),
                adaptive=(str(hyper["adaptive"]).lower() == "true"),
                initial_velocity=hyper["initial_velocity"],
                max_velocity_rate=float(hyper["max_velocity_rate"]),
                pertube_best=(str(hyper["pertube_best"]).lower() == "true")
            )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )

            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]

            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "PSO terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "PSO optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name == 'GA' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)

            hyper = config["hyperparameters"]

            print("GA hyperparameters:")
            print(json.dumps(hyper, indent=4))
            sampling = None
            selection = None
            if str(hyper["sampling"]).lower() == "real_random":
                sampling = FloatRandomSampling()
            if str(hyper["crossover"]).lower() == "sbx":
                crossover = SBX()
            if str(hyper["mutation"]).lower() == "pm":
                mutation = PM()

            algorithm = GA( pop_size=int(hyper["pop_size"]),
                            sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            eliminate_duplicates= (str(hyper["eliminate_duplicates"]).lower() == "true"),
                            n_offsprings=int(hyper["offsprings"])
                          )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )

            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]

            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "GA terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "GA optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name == 'DE' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)

            hyper = config["hyperparameters"]

            print("DE hyperparameters:")
            print(json.dumps(hyper, indent=4))


            algorithm = DE(pop_size = int(hyper["pop_size"]),
                           n_offsprings=  int(hyper["n_offsprings"] )
                          ,sampling= FloatRandomSampling() )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )

            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]

            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "DE terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "DE optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name ==  'NelderMead' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)

            hyper = config["hyperparameters"]

            print("NelderMead hyperparameters:")
            print(json.dumps(hyper, indent=4))
            #ERROR: Faltaba especificarlo aqui en el ciclo
            algorithm = NelderMead(initialize_simplex= float(hyper["simplex"]))

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )

            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]

            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "NelderMead terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "NelderMead optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name == 'PatternSearch' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("PatternSearch hyperparameters:")
            print(json.dumps(hyper, indent=4))
            
            algorithm = PatternSearch(init_delta= float(hyper["delta"]),
                                     init_rho= float(hyper["rho"]),
                                     step_size= float(hyper["step_size"]),)
            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "PatternSearch terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "PatternSearch optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name == 'BRKGA' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("BRKGA hyperparameters:")
            print(json.dumps(hyper, indent=4))
            if str(hyper["sampling"]).lower() == "real_random":
                sampling = FloatRandomSampling()
            
             
            algorithm = BRKGA(n_elites =  int(hyper["n_elites"]),
                              n_offsprings = int(hyper["n_offsprings"]),
                              n_mutants = int(hyper["n_mutants"]), 
                              bias= float(hyper["crossover_bias"]),
                              sampling = sampling)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "BRKGA terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "BRKGA optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name ==  'ES' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("ES hyperparameters:")
            print(json.dumps(hyper, indent=4))


            if str(hyper["sampling"]).lower() == "real_random":
                sampling = FloatRandomSampling()
            
            # ERROR: NO SE ESTABA PASANDO EL VALOR DE N_OFFSPRINGS EN EL JSON
            algorithm = ES( n_offsprings= int(hyper["n_offsprings"]),
                            pop_size= int(hyper["pop_size"]),
                            rule =  float(hyper["rule"]),
                            phi = float(hyper["phi"]),
                            gamma = float(hyper["gamma"]))

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "ES terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "ES optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name ==  'SRES' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("SRES hyperparameters:")
            print(json.dumps(hyper, indent=4))

            #ERROR: SOLO HAY QUE PASAS PF  
            algorithm = SRES(PF =  float(hyper["pf"]) )
            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "SRES terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "SRES optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name ==  'ISRES' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("ISRES hyperparameters:")
            print(json.dumps(hyper, indent=4))

             
            algorithm = ISRES(n_offsprings= int(hyper["n_offsprings"]), 
                             rule= float(hyper["rule"]), 
                             gamma=float(hyper["gamma"]), 
                             alpha=float(hyper["alpha"]))
            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "ISRES terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "ISRESoptimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name ==  'CMAES' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("CMAES hyperparameters:")
            print(json.dumps(hyper, indent=4))

            # ERROR: AQUI FALTA PASAR UN ARGUMENTO 
            x0_array, x0_records = initialize_CMAES(
                config=config,
                n_init=int(hyper["n_init"]),
                seed=int(hyper["seed"])
            )

            if x0_array is None or len(x0_array) == 0:
                raise RuntimeError("No se pudo generar ningún x0 factible para CMAES.")

            # CMA-ES necesita un solo vector inicial, no una matriz de n_init puntos.
            x0 = np.asarray(x0_array[0], dtype=float).reshape(-1)

            opts = {
                "verb_disp": 0,
                "verbose": -9,
                "verb_log": 0
            }

            if "pop_size" in hyper and str(hyper["pop_size"]).strip() != "":
                opts["popsize"] = int(hyper["pop_size"])

            algorithm = CMAES(
                x0=x0,
                sigma=float(hyper["sigma"]),
                normalize=True,
                opts=opts
            )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "CMAES terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "CMAES optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name ==  'G3PCX' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("G3PCX hyperparameters:")
            print(json.dumps(hyper, indent=4))

            if str(hyper["sampling"]).lower() == "real_random":
                sampling = FloatRandomSampling() 
            algorithm = G3PCX(pop_size = int(hyper["pop_size"]),
                              n_offsprings = int(hyper["n_offsprings"]),
                              n_parents = int(hyper["n_parents"]),
                              family_size = int(hyper["family_size"]),
                              sampling = sampling  )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "G3PCX terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "G3PCX optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name ==  'NRBO' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblem(config, aspen)
            hyper = config["hyperparameters"]
            print("NRBO hyperparameters:")
            print(json.dumps(hyper, indent=4))

            algorithm = NRBO(pop_size = int(hyper["pop_size"]),
                             deciding_factor= float(hyper["deciding_factor"]),
                             max_iteration= int(hyper["max_iteration"]), )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(hyper["generations"])),
                seed=int(hyper["seed"]),
                verbose=False
            )
            valid_records = [
                row for row in problem.history
                if row["feasible"] and row["aspen_converged"] and np.isfinite(row["objective"])
            ]
            if len(valid_records) == 0:
                results = {
                    "status": "error",
                    "message": "NRBO terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                    "best_x": None,
                    "best_objective": None,
                    "outputs": None,
                    "evaluations": problem.history,
                }
            else:
                best_record = min(valid_records, key=lambda r: r["objective"])
                results = {
                    "status": "success",
                    "message": "NRBO optimization finished.",
                    "best_x": best_record["x"],
                    "best_objective": float(best_record["objective"]),
                    "outputs": best_record["outputs"],
                    "evaluations": problem.history,
                }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, problem.history)
        elif algorithm_name == 'SBOA' and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            optimizer = SBOA(config, aspen)
            results, history = optimizer.run()
            #results, history = run_sboa_optimization(config, aspen)

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_mono(config, results_path, history)        
        elif algorithm_name == "Discrete Bayesian Optimization" and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            optimizer = DiscreteBOBotorch(config, aspen)
            results, history = optimizer.run()
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, history)
        elif algorithm_name == "Vanilla Bayesian Optimization" and problem_type == "single_objective":
            aspen = open_aspen_file(config["aspen_file"])
            optimizer = VanillaBOBotorch(config, aspen)
            results, history = optimizer.run()
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, history)
        elif algorithm_name == "Trust Region Bayesian Optimization" and problem_type == "single_objective":
            _, _, only_continuous, _, _ = get_variable_type_info(config)
            if not only_continuous:
                raise RuntimeError(
                    "Trust Region Bayesian Optimization solo está habilitado para variables continuas. "
                    "Cambia todas las variables de entrada a Continuous."
                )

            aspen = open_aspen_file(config["aspen_file"])
            optimizer = TrustRegionBOBotorch(config, aspen)
            results, history = optimizer.run()
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_mono(config, results_path, history)
        elif algorithm_name == "MORPHBO" and problem_type == "single_objective":
            unique_list, unique_list_lower, only_continuous, only_discrete, mixed = get_variable_type_info(config)

            allowed_types = {"continuous", "discrete", "integer"}
            unsupported = [v for v in unique_list_lower if v not in allowed_types]

            if len(unsupported) > 0:
                raise RuntimeError(
                    "MORPH-BO-F monoobjetivo solo acepta variables Continuous y Discrete. "
                    f"Tipos encontrados no soportados: {unsupported}"
                )

            aspen = open_aspen_file(config["aspen_file"])
            optimizer = MorphBOBotorch(config, aspen)

            results, history = optimizer.run()

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_mono(config, results_path, history)
        elif algorithm_name == "NSGA-II" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            # Revisar si el problema solo tiene variables continuas
            list_variable_types = []

            for input_var in config["inputs"]:
                variable_type = input_var.get("variable_type", "Continuous")
                list_variable_types.append(variable_type)
            unique_list  = list(set( list_variable_types ))
            # EL problema es unicamente continuo 
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]
            pop_size = int(hyper.get("pop_size", 40))
            n_gen = int(hyper.get("n_gen", hyper.get("generations", 50)))
            seed = int(hyper.get("seed", 42))
            if len(unique_list) == 1 and unique_list[0].lower() == "continuous":
                    algorithm = NSGA2(pop_size=pop_size,
                                      eliminate_duplicates=True)
            elif len(unique_list) == 1 and unique_list[0].lower() == "discrete":
                    algorithm = NSGA2(pop_size=pop_size,
                                      eliminate_duplicates=True)
            elif len(unique_list) > 1: 
                    n_offsprings =  int(hyper.get("n_offsprings", min(10, pop_size)))
                    algorithm = NSGA2(pop_size=pop_size, n_offsprings= n_offsprings, sampling=MixedVariableSampling(),
                                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                                eliminate_duplicates=MixedVariableDuplicateElimination())
            # Optimiza
            res = minimize( problem,
                                    algorithm,
                                    ("n_gen", n_gen),
                                    seed=seed,
                                    verbose=False,
                                    save_history=False
                                )
            pareto_F_raw = problem.restore_raw_objectives(res.F)

            results = {
                        "algorithm": config["algorithm"],
                        "problem_type": config["problem_type"],
                        "objective_names": problem.objective_names,

                        # Valores usados por pymoo. Todos son de minimización.
                        "pareto_F_minimized": res.F.tolist() if res.F is not None else [],

                        # Valores reales. Aquí los objetivos Max regresan a su signo original.
                        "pareto_F_raw": pareto_F_raw.tolist() if pareto_F_raw is not None else [],

                        # Variables de decisión del frente de Pareto.
                        "pareto_X": res.X.tolist() if res.X is not None else [],

                        # Todas las evaluaciones hechas en Aspen.
                        "history": problem.history
                    }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_multi(config, results_path, problem.history)
        elif algorithm_name == "NSGA-III" and problem_type == "multi_objective":

            aspen = open_aspen_file(config["aspen_file"])

            list_variable_types = []

            for input_var in config["inputs"]:
                variable_type = input_var.get("variable_type", "Continuous")
                list_variable_types.append(variable_type)

            unique_list = list(set(list_variable_types))
            unique_list_lower = [v.lower() for v in unique_list]

            problem = AspenProblemMOBO(config, aspen)

            hyper = config["hyperparameters"]

            n_gen = int(hyper.get("n_gen", hyper.get("generations", 50)))
            seed = int(hyper.get("seed", 42))

            # ==========================================================
            # Reference directions para NSGA-III
            # ==========================================================
            ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)

            # Recomendación: en NSGA-III usa pop_size = len(ref_dirs)
            # para que la población coincida con las direcciones de referencia.
            pop_size = len(ref_dirs)

            # ==========================================================
            # Caso 1: variables continuas
            # ==========================================================
            if len(unique_list) == 1 and unique_list_lower[0] == "continuous":
                algorithm = NSGA3(
                    ref_dirs=ref_dirs,
                    pop_size=pop_size,
                    eliminate_duplicates=True
                )

            elif len(unique_list) == 1 and unique_list_lower[0] == "discrete":
                algorithm = NSGA3(
                    ref_dirs=ref_dirs,
                    pop_size=pop_size,
                    eliminate_duplicates=True
                )
            elif len(unique_list) > 1:
                n_offsprings = int(hyper.get("n_offsprings", min(10, pop_size)))
                algorithm = NSGA3(
                    ref_dirs=ref_dirs,
                    pop_size=pop_size,
                    n_offsprings=n_offsprings,
                    sampling=MixedVariableSampling(),
                    mating=MixedVariableMating(
                        eliminate_duplicates=MixedVariableDuplicateElimination()
                    ),
                    eliminate_duplicates=MixedVariableDuplicateElimination()
                )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
                )

            pareto_F_raw = problem.restore_raw_objectives(res.F) if res.F is not None else None

            results = {
                "algorithm": config["algorithm"],
                "problem_type": config["problem_type"],
                "objective_names": problem.objective_names,

                # Valores usados por pymoo. Todos son de minimización.
                "pareto_F_minimized": res.F.tolist() if res.F is not None else [],

                # Valores reales. Aquí los objetivos Max regresan a su signo original.
                "pareto_F_raw": pareto_F_raw.tolist() if pareto_F_raw is not None else [],

                # Variables de decisión del frente de Pareto.
                "pareto_X": res.X.tolist() if res.X is not None else [],

                # Reference directions usadas por NSGA-III.
                "ref_dirs": ref_dirs.tolist(),

                # Todas las evaluaciones hechas en Aspen.
                "history": problem.history
            }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_multi(config, results_path, problem.history)
        elif algorithm_name == "R-NSGA-II" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)

            pop_size = int(hyper.get("pop_size", 40))
            n_gen = int(hyper.get("n_gen", hyper.get("generations", 50)))
            seed = int(hyper.get("seed", 42))

            ref_points = build_reference_points_for_rnsga(config, problem)
            weights = build_weights_for_rnsga(config, problem)

            kwargs = {
                "ref_points": ref_points,
                "pop_size": pop_size,
                "epsilon": hyper_float(hyper, "epsilon", 0.001),
                "normalization": hyper_text(hyper, "normalization", "front"),
                "extreme_points_as_reference_points": str_to_bool(
                    hyper.get("extreme_points_as_reference_points", False),
                    default=False
                ),
                "eliminate_duplicates": True
            }

            if weights is not None:
                kwargs["weights"] = weights

            if mixed:
                n_offsprings = int(hyper.get("n_offsprings", min(10, pop_size)))
                kwargs.update({
                    "n_offsprings": n_offsprings,
                    "sampling": MixedVariableSampling(),
                    "mating": MixedVariableMating(
                        eliminate_duplicates=MixedVariableDuplicateElimination()
                    ),
                    "eliminate_duplicates": MixedVariableDuplicateElimination()
                })

            algorithm = RNSGA2(**kwargs)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            results = make_multiobjective_results(
                config,
                problem,
                res,
                extra={
                    "ref_points": ref_points.tolist(),
                    "weights": weights.tolist() if weights is not None else None
                }
            )

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_multi(config, results_path, problem.history)
        elif algorithm_name == "R-NSGA-III" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)

            n_gen = int(hyper.get("n_gen", hyper.get("generations", 50)))
            seed = int(hyper.get("seed", 42))

            ref_points = build_reference_points_for_rnsga(config, problem)
            pop_per_ref_point = hyper_int(hyper, "pop_per_ref_point", hyper_value(hyper, "pop_size", 50))
            mu = hyper_float(hyper, "mu", 0.05)

            kwargs = {
                "ref_points": ref_points,
                "pop_per_ref_point": pop_per_ref_point,
                "mu": mu,
                "eliminate_duplicates": True
            }

            if mixed:
                n_offsprings = int(hyper.get("n_offsprings", min(10, pop_per_ref_point)))
                kwargs.update({
                    "n_offsprings": n_offsprings,
                    "sampling": MixedVariableSampling(),
                    "mating": MixedVariableMating(
                        eliminate_duplicates=MixedVariableDuplicateElimination()
                    ),
                    "eliminate_duplicates": MixedVariableDuplicateElimination()
                })

            algorithm = RNSGA3(**kwargs)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            # pymoo guarda las reference directions internas en survival.ref_dirs.
            ref_dirs_used = None
            try:
                ref_dirs_used = res.algorithm.survival.ref_dirs
            except Exception:
                ref_dirs_used = None

            results = make_multiobjective_results(
                config,
                problem,
                res,
                extra={
                    "ref_points": ref_points.tolist(),
                    "pop_per_ref_point": pop_per_ref_point,
                    "mu": mu,
                    "ref_dirs": ref_dirs_used.tolist() if ref_dirs_used is not None else []
                }
            )

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_multi(config, results_path, problem.history)
        elif algorithm_name == "MOPSO-CD" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
            if not only_continuous:
                print(
                    "WARNING: MOPSO-CD trabaja mejor con variables continuas. "
                    "Si usas variables discretas, AspenProblemMOBO debe redondearlas/repararlas internamente.",
                    flush=True
                )

            pop_size = int(hyper.get("pop_size", 40))
            n_gen = int(hyper.get("n_gen", hyper.get("generations", 50)))
            seed = int(hyper.get("seed", 42))

            algorithm = MOPSO_CD(
                pop_size=pop_size,
                w=float(hyper.get("w", 0.6)),
                c1=float(hyper.get("c1", 2.0)),
                c2=float(hyper.get("c2", 2.0)),
                max_velocity_rate=float(hyper.get("max_velocity_rate", 0.5)),
                archive_size=int(hyper.get("archive_size", max(2 * pop_size, 100))),
                sampling=FloatRandomSampling()
            )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            results = make_multiobjective_results(
                config,
                problem,
                res,
                extra={
                    "pop_size": pop_size,
                    "archive_size": int(hyper.get("archive_size", max(2 * pop_size, 100)))
                }
            )

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_multi(config, results_path, problem.history)
        elif algorithm_name == "CMOPSO" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
            if not only_continuous:
                print(
                    "WARNING: CMOPSO trabaja mejor con variables continuas. "
                    "Si usas variables discretas, AspenProblemMOBO debe redondearlas/repararlas internamente.",
                    flush=True
                )

            pop_size = int(hyper.get("pop_size", 40))
            n_gen = int(hyper.get("n_gen", hyper.get("generations", 50)))
            seed = int(hyper.get("seed", 42))

            algorithm = CMOPSO(
                pop_size=pop_size,
                max_velocity_rate=float(hyper.get("max_velocity_rate", 0.2)),
                elite_size=int(hyper.get("elite_size", 10)),
                initial_velocity=str(hyper.get("initial_velocity", "random")),
                mutation_rate=float(hyper.get("mutation_rate", 0.5)),
                sampling=FloatRandomSampling()
            )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            results = make_multiobjective_results(
                config,
                problem,
                res,
                extra={
                    "pop_size": pop_size,
                    "elite_size": int(hyper.get("elite_size", 10))
                }
            )

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_multi(config, results_path, problem.history)
        elif algorithm_name == "U-NSGA-III" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)

            n_gen = hyper_int(hyper, "n_gen", hyper_value(hyper, "generations", 50))
            seed = hyper_int(hyper, "seed", 42)
            ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)
            pop_size = hyper_int(hyper, "pop_size", len(ref_dirs))

            kwargs = {
                "ref_dirs": ref_dirs,
                "pop_size": pop_size,
                "eliminate_duplicates": str_to_bool(hyper_value(hyper, "eliminate_duplicates", True), default=True)
            }

            if "n_offsprings" in hyper and str(hyper.get("n_offsprings", "")).strip() != "":
                kwargs["n_offsprings"] = hyper_int(hyper, "n_offsprings", min(10, pop_size))

            if mixed:
                kwargs.update({
                    "sampling": MixedVariableSampling(),
                    "mating": MixedVariableMating(
                        eliminate_duplicates=MixedVariableDuplicateElimination()
                    ),
                    "eliminate_duplicates": MixedVariableDuplicateElimination()
                })

            algorithm = UNSGA3(**kwargs)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            save_multiobjective_run(
                config,
                results_path,
                problem,
                res,
                extra={
                    "ref_dirs": ref_dirs.tolist(),
                    "pop_size": pop_size
                }
            )
        elif algorithm_name == "MOEA/D" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
            warn_if_not_continuous_for_moo("MOEA/D", only_continuous)

            n_gen = hyper_int(hyper, "n_gen", hyper_value(hyper, "generations", 50))
            seed = hyper_int(hyper, "seed", 42)
            ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)
            n_neighbors = hyper_int(hyper, "n_neighbors", 20)
            prob_neighbor_mating = hyper_float(hyper, "prob_neighbor_mating", 0.9)

            algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=n_neighbors,
                prob_neighbor_mating=prob_neighbor_mating,
                sampling=FloatRandomSampling()
            )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            save_multiobjective_run(
                config,
                results_path,
                problem,
                res,
                extra={
                    "ref_dirs": ref_dirs.tolist(),
                    "n_neighbors": n_neighbors,
                    "prob_neighbor_mating": prob_neighbor_mating
                }
            )
        elif algorithm_name == "AGE-MOEA" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
            warn_if_not_continuous_for_moo("AGE-MOEA", only_continuous)

            pop_size = hyper_int(hyper, "pop_size", 40)
            n_gen = hyper_int(hyper, "n_gen", hyper_value(hyper, "generations", 50))
            seed = hyper_int(hyper, "seed", 42)

            kwargs = {
                "pop_size": pop_size,
                "eliminate_duplicates": str_to_bool(hyper_value(hyper, "eliminate_duplicates", True), default=True)
            }

            if "n_offsprings" in hyper and str(hyper.get("n_offsprings", "")).strip() != "":
                kwargs["n_offsprings"] = hyper_int(hyper, "n_offsprings", min(10, pop_size))

            algorithm = AGEMOEA(**kwargs)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            save_multiobjective_run(
                config,
                results_path,
                problem,
                res,
                extra={"pop_size": pop_size}
            )
        elif algorithm_name == "AGE-MOEA2" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
            warn_if_not_continuous_for_moo("AGE-MOEA2", only_continuous)

            pop_size = hyper_int(hyper, "pop_size", 40)
            n_gen = hyper_int(hyper, "n_gen", hyper_value(hyper, "generations", 50))
            seed = hyper_int(hyper, "seed", 42)

            kwargs = {
                "pop_size": pop_size,
                "eliminate_duplicates": str_to_bool(hyper_value(hyper, "eliminate_duplicates", True), default=True)
            }

            if "n_offsprings" in hyper and str(hyper.get("n_offsprings", "")).strip() != "":
                kwargs["n_offsprings"] = hyper_int(hyper, "n_offsprings", min(10, pop_size))

            algorithm = AGEMOEA2(**kwargs)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            save_multiobjective_run(
                config,
                results_path,
                problem,
                res,
                extra={"pop_size": pop_size}
            )
        elif algorithm_name == "RVEA" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
            warn_if_not_continuous_for_moo("RVEA", only_continuous)

            n_gen = hyper_int(hyper, "n_gen", hyper_value(hyper, "generations", 50))
            seed = hyper_int(hyper, "seed", 42)
            ref_dirs = build_reference_directions_for_nsga3(config, problem, seed)
            pop_size = hyper_int(hyper, "pop_size", len(ref_dirs))
            alpha = hyper_float(hyper, "alpha", 2.0)
            adapt_freq = hyper_float(hyper, "adapt_freq", 0.1)

            kwargs = {
                "ref_dirs": ref_dirs,
                "alpha": alpha,
                "adapt_freq": adapt_freq,
                "pop_size": pop_size,
                "eliminate_duplicates": str_to_bool(hyper_value(hyper, "eliminate_duplicates", True), default=True)
            }

            if "n_offsprings" in hyper and str(hyper.get("n_offsprings", "")).strip() != "":
                kwargs["n_offsprings"] = hyper_int(hyper, "n_offsprings", min(10, pop_size))

            algorithm = RVEA(**kwargs)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            save_multiobjective_run(
                config,
                results_path,
                problem,
                res,
                extra={
                    "ref_dirs": ref_dirs.tolist(),
                    "pop_size": pop_size,
                    "alpha": alpha,
                    "adapt_freq": adapt_freq
                }
            )
        elif algorithm_name == "SMS-EMOA" and problem_type == "multi_objective":
            aspen = open_aspen_file(config["aspen_file"])
            problem = AspenProblemMOBO(config, aspen)
            hyper = config["hyperparameters"]

            _, _, only_continuous, only_discrete, mixed = get_variable_type_info(config)
            warn_if_not_continuous_for_moo("SMS-EMOA", only_continuous)

            pop_size = hyper_int(hyper, "pop_size", 40)
            n_gen = hyper_int(hyper, "n_gen", hyper_value(hyper, "generations", 50))
            seed = hyper_int(hyper, "seed", 42)
            normalize = str_to_bool(hyper_value(hyper, "normalize", True), default=True)

            kwargs = {
                "pop_size": pop_size,
                "normalize": normalize,
                "eliminate_duplicates": str_to_bool(hyper_value(hyper, "eliminate_duplicates", True), default=True)
            }

            if "n_offsprings" in hyper and str(hyper.get("n_offsprings", "")).strip() != "":
                kwargs["n_offsprings"] = hyper_int(hyper, "n_offsprings", 1)

            algorithm = SMSEMOA(**kwargs)

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=False,
                save_history=False
            )

            save_multiobjective_run(
                config,
                results_path,
                problem,
                res,
                extra={
                    "pop_size": pop_size,
                    "normalize": normalize
                }
            )
        elif algorithm_name == "Vanilla M.O. Bayesian Optimization" and problem_type == "multi_objective":
            _, _, only_continuous, _, _ = get_variable_type_info(config)
            if not only_continuous:
                raise RuntimeError(
                    "Vanilla M.O. Bayesian Optimization solo está habilitado para variables continuas. "
                    "Cambia todas las variables de entrada a Continuous."
                )

            aspen = open_aspen_file(config["aspen_file"])
            optimizer = ContinuousMOBOBotorch(config, aspen)
            results, history = optimizer.run()
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            write_history_csv_multi(config, results_path, history)
        elif algorithm_name == "MORPHBO" and problem_type == "multi_objective":
            unique_list, unique_list_lower, only_continuous, only_discrete, mixed = get_variable_type_info(config)

            allowed_types = {"continuous", "discrete", "integer"}
            unsupported = [v for v in unique_list_lower if v not in allowed_types]

            if len(unsupported) > 0:
                raise RuntimeError(
                    "MORPHBO multiobjetivo solo acepta variables Continuous y Discrete. "
                    f"Tipos encontrados no soportados: {unsupported}"
                )

            aspen = open_aspen_file(config["aspen_file"])
            optimizer = MorphMOBOBotorch(config, aspen)
            results, history = optimizer.run()

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            write_history_csv_multi(config, results_path, history)

    except Exception as e:
        results = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        raise

    finally:
        close_aspen(aspen)



if __name__ == "__main__":
    main()





