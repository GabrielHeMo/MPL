"""Shared optimizer configuration helpers extracted from optimizer_runner.py."""
from __future__ import annotations

import json
import re
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

from Evaluation.remote_aspen_problem_functions import RemoteAspenProblem, RemoteAspenProblemMOBO


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
    hyper = config['hyperparameters']
    n_obj = int(getattr(problem, 'n_obj', len(config['objectives'])))
    ref_dir_method = str(hyper.get('ref_dir_method', 'energy')).lower()
    if ref_dir_method == 'energy':
        n_ref_dirs = int(hyper.get('n_ref_dirs', hyper.get('pop_size', default_n_ref_dirs(n_obj))))
        ref_dirs = get_reference_directions('energy', n_obj, n_ref_dirs, seed=seed)
        return ref_dirs
    elif ref_dir_method in ['das-dennis', 'das_dennis', 'uniform']:
        n_partitions = int(hyper.get('n_partitions', 12))
        ref_dirs = get_reference_directions('das-dennis', n_obj, n_partitions=n_partitions)
        return ref_dirs
    else:
        raise ValueError(f'Unknown ref_dir_method for NSGA-III: {ref_dir_method}')

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
    if text in ['true', '1', 'yes', 'y', 'si', 'sí']:
        return True
    if text in ['false', '0', 'no', 'n']:
        return False
    return default

def parse_numeric_vector_or_matrix(raw, name='value'):
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
        if text == '':
            return None
        try:
            return np.array(json.loads(text), dtype=float)
        except Exception:
            pass
        rows = []
        for row_text in text.replace('|', ';').split(';'):
            row_text = row_text.strip()
            if row_text == '':
                continue
            values = [v for v in re.split('[\\s,]+', row_text) if v.strip() != '']
            rows.append([float(v) for v in values])
        if len(rows) == 0:
            raise ValueError(f'No se pudo interpretar {name}: {raw}')
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
    hyper = config.get('hyperparameters', {}) or {}
    n_obj = int(getattr(problem, 'n_obj', len(config.get('objectives', []))))
    raw = hyper.get('ref_points', None) or hyper.get('reference_points', None) or hyper.get('aspiration_points', None)
    ref_points = parse_numeric_vector_or_matrix(raw, name='ref_points')
    if ref_points is None:
        raise ValueError('R-NSGA-II/R-NSGA-III requieren ref_points. Ejemplo para 2 objetivos: [[0.5, 0.2], [0.1, 0.6]] o como texto: 0.5,0.2;0.1,0.6. Cada fila debe tener el mismo número de columnas que objetivos.')
    if ref_points.ndim == 1:
        ref_points = ref_points.reshape(1, -1)
    if ref_points.ndim != 2:
        raise ValueError('ref_points debe ser una matriz 2D: filas=puntos, columnas=objetivos.')
    if ref_points.shape[1] != n_obj:
        raise ValueError(f'ref_points tiene {ref_points.shape[1]} columnas, pero el problema tiene {n_obj} objetivos.')
    return ref_points.astype(float)

def build_weights_for_rnsga(config, problem):
    """
    Pesos opcionales para R-NSGA-II.
    Si no se mandan desde C#, se usa None y pymoo aplica su valor por defecto.
    """
    hyper = config.get('hyperparameters', {}) or {}
    n_obj = int(getattr(problem, 'n_obj', len(config.get('objectives', []))))
    weights = parse_numeric_vector_or_matrix(hyper.get('weights', None), name='weights')
    if weights is None:
        return None
    weights = np.array(weights, dtype=float).reshape(-1)
    if len(weights) != n_obj:
        raise ValueError(f'weights tiene {len(weights)} valores, pero el problema tiene {n_obj} objetivos.')
    return weights

def get_variable_type_info(config):
    list_variable_types = []
    for input_var in config.get('inputs', []):
        variable_type = input_var.get('variable_type', 'Continuous')
        list_variable_types.append(str(variable_type))
    unique_list = list(set(list_variable_types))
    unique_list_lower = [v.lower() for v in unique_list]
    only_continuous = len(unique_list_lower) == 1 and unique_list_lower[0] == 'continuous'
    only_discrete = len(unique_list_lower) == 1 and unique_list_lower[0] == 'discrete'
    mixed = len(unique_list_lower) > 1
    return (unique_list, unique_list_lower, only_continuous, only_discrete, mixed)

def hyper_value(hyper, key, default=None):
    """Lee un hiperparámetro enviado desde C# y permite usar defaults si viene vacío."""
    value = hyper.get(key, None)
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == '':
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
        print(f'WARNING: {algorithm_name} trabaja mejor con variables continuas. Si usas variables discretas, el bridge de Aspen en C# las redondeará antes de evaluarlas.', flush=True)

def require_csharp_backend(config):
    evaluation_backend = str(config.get('evaluation_backend', '')).strip().lower()
    if evaluation_backend != 'csharp_jsonl':
        raise RuntimeError(f"This MPL build requires the C# -> Aspen JSONL evaluation backend. Received evaluation_backend='{evaluation_backend or 'missing'}'.")

def create_single_objective_problem(config):
    require_csharp_backend(config)
    return (RemoteAspenProblem(config), None)

def create_multiobjective_problem(config):
    require_csharp_backend(config)
    return (RemoteAspenProblemMOBO(config), None)
