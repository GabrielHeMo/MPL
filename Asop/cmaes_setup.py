import numpy as np
import math
# ==========================================================
# Compatibility patch for NumPy 2.x
# Some versions of pycma still use old NumPy aliases.
# ==========================================================
if not hasattr(np, "Inf"):
    np.Inf = np.inf

if not hasattr(np, "Infinity"):
    np.Infinity = np.inf

if not hasattr(np, "NINF"):
    np.NINF = -np.inf

if not hasattr(np, "NaN"):
    np.NaN = np.nan

if not hasattr(np, "float_"):
    np.float_ = np.float64

import cma
import cma.transformations as cma_transformations
import cma.evolution_strategy as cma_evolution_strategy


def _numpy2_array_compat(
    obj,
    dtype=None,
    copy=True,
    order=None,
    subok=False,
    ndmin=0,
    like=None
):
    """
    Compatibilidad para pycma con NumPy 2.x.

    En NumPy 2, np.array(obj, copy=False) puede lanzar:
    ValueError: Unable to avoid copy while creating an array as requested.

    Para el caso copy=False, usamos np.asarray(...), que es lo que
    recomienda NumPy para permitir copia cuando sea necesario.
    """

    if copy is False:
        arr = np.asarray(obj, dtype=dtype, order=order)

        if ndmin is not None and ndmin > 0 and arr.ndim < ndmin:
            arr = np.array(arr, ndmin=ndmin)

        return arr

    kwargs = {}

    if dtype is not None:
        kwargs["dtype"] = dtype

    if copy is not None:
        kwargs["copy"] = copy

    if order is not None:
        kwargs["order"] = order

    if subok is not None:
        kwargs["subok"] = subok

    if ndmin is not None:
        kwargs["ndmin"] = ndmin

    if like is not None:
        kwargs["like"] = like

    return np.array(obj, **kwargs)

# Este es el parche importante:
# pycma usa una variable global llamada "array" dentro de cma.transformations.
# La reemplazamos solo en ese módulo, no globalmente en NumPy.
cma_transformations.array = _numpy2_array_compat
cma_evolution_strategy.array = _numpy2_array_compat

from pymoo.algorithms.base.local import LocalSearch
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.normalization import ZeroToOneNormalization, NoNormalization
from pymoo.util.optimum import filter_optimum

# FUNCIONES AUXILIARES PARA CMAES 

def round_to_valid_x(config, x, lb, ub):
    """
    Ajusta x a los límites y redondea variables discretas.
    """
    x = np.array(x, dtype=float).reshape(-1)
    x = np.minimum(np.maximum(x, lb), ub)

    for i, input_var in enumerate(config["inputs"]):
        vtype = input_var.get("variable_type", "Continuous").lower()

        if vtype == "discrete":
            x[i] = round(x[i])
            x[i] = min(max(x[i], lb[i]), ub[i])

    return x.astype(float)


def point_key(config, x, lb, ub):
    """
    Genera una clave única para evitar repetir puntos.
    """
    x = round_to_valid_x(config, x, lb, ub)
    key = []

    for i, input_var in enumerate(config["inputs"]):
        vtype = input_var.get("variable_type", "Continuous").lower()

        if vtype == "discrete":
            key.append(int(round(x[i])))
        else:
            key.append(round(float(x[i]), 10))

    return tuple(key)


def input_context(config, x):
    """
    Construye un diccionario tipo {'x1': valor, 'x2': valor, ...}
    para evaluar restricciones.
    """
    ctx = {}

    for i, (input_var, value) in enumerate(zip(config["inputs"], x)):
        tag = input_var.get("tag") or input_var.get("alias") or f"x{i + 1}"
        ctx[tag] = float(value)

    return ctx


def normalize_key(d, *names, default=None):
    """
    Permite leer constraints aunque C# mande nombres como:
    LeftSide, left_side, leftSide, RightSide, etc.
    """
    if not isinstance(d, dict):
        return default

    lower_map = {
        str(k).lower().replace("_", ""): k
        for k in d.keys()
    }

    for name in names:
        key = lower_map.get(str(name).lower().replace("_", ""))
        if key is not None:
            return d[key]

    return default


def resolve_value(token, ctx):
    """
    Convierte un lado de la restricción a número.
    Puede ser una constante, un tag tipo x1, o una expresión tipo x1 + x2.
    """
    if isinstance(token, (int, float)):
        return float(token)

    text = str(token).strip()

    if text in ctx:
        return float(ctx[text])

    try:
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

    return float(eval(text, safe_globals, safe_locals))


def compare_values(left, operator, right, tol=1e-9):
    """
    Compara dos valores usando el operador indicado.
    """
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


def check_input_constraints(config, x):
    """
    Evalúa restricciones de entrada antes de generar el x0.
    No corre Aspen.
    """
    constraints = config.get("constraints", []) or []
    ctx = input_context(config, x)
    violations = []

    for i, constraint in enumerate(constraints, start=1):
        ctype = str(
            normalize_key(
                constraint,
                "type",
                "constraint_type",
                default="Hard"
            )
        ).lower()

        if ctype == "soft":
            continue

        left_token = normalize_key(
            constraint,
            "left_side",
            "leftside",
            "left",
            default=None
        )

        operator = normalize_key(
            constraint,
            "operator",
            "op",
            default=None
        )

        right_token = normalize_key(
            constraint,
            "right_side",
            "rightside",
            "right",
            default=None
        )

        if left_token in [None, ""] or operator in [None, ""] or right_token in [None, ""]:
            continue

        try:
            left_value = resolve_value(left_token, ctx)
            right_value = resolve_value(right_token, ctx)
            ok = compare_values(left_value, operator, right_value)

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


def satisfies_input_constraints(config, x):
    """
    Regresa True si x cumple las restricciones Hard.
    """
    ok, _ = check_input_constraints(config, x)
    return ok


def random_unobserved(config, observed_keys, dim, lb, ub, rng, max_tries=5000):
    """
    Genera un punto aleatorio dentro de bounds, no repetido y que cumpla restricciones.
    No evalúa Aspen.
    """
    for _ in range(max_tries):
        xn = rng.random(dim)

        # Desnormalizar de [0, 1] a [lb, ub]
        x = lb + xn * (ub - lb)

        # Ajustar límites y discretas
        x = round_to_valid_x(config, x, lb, ub)

        # Crear llave para evitar duplicados
        key = point_key(config, x, lb, ub)

        if key in observed_keys:
            continue

        if not satisfies_input_constraints(config, x):
            continue

        return x

    return None


def initialize_CMAES(config, n_init=20, seed=42):
    """
    Genera puntos iniciales para CMA-ES sin evaluarlos en Aspen.

    Retorna:
    - candidates_array: matriz numpy de tamaño (n_init, dim)
    - candidates_records: lista con diccionarios por candidato
    """

    input_tags = [
        v.get("tag") or v.get("alias") or f"x{i + 1}"
        for i, v in enumerate(config.get("inputs", []))
    ]

    lb = np.array(
        [float(v["lower_bound"]) for v in config["inputs"]],
        dtype=float
    )

    ub = np.array(
        [float(v["upper_bound"]) for v in config["inputs"]],
        dtype=float
    )

    dim = len(input_tags)

    rng = np.random.default_rng(seed)
    observed_keys = set()
    candidates_records = []

    while len(candidates_records) < n_init:
        x0 = random_unobserved(
            config=config,
            observed_keys=observed_keys,
            dim=dim,
            lb=lb,
            ub=ub,
            rng=rng
        )

        if x0 is None:
            break

        key = point_key(config, x0, lb, ub)
        observed_keys.add(key)

        record = {
            "candidate": len(candidates_records) + 1,
            "x": {
                input_tags[i]: float(x0[i])
                for i in range(dim)
            },
            "x_array": x0.tolist(),
            "key": key
        }

        candidates_records.append(record)

    candidates_array = np.array(
        [record["x_array"] for record in candidates_records],
        dtype=float
    )

    return candidates_array, candidates_records


class CMAES(LocalSearch):
    """
    CMA-ES compatible con NumPy 2.x.

    Diferencia principal contra pymoo.CMAES:
    - No usa los bounds internos de pycma.
    - Aplica clipping manual antes de evaluar el problema.
    - Usa ask/tell directamente.
    """

    def __init__(
        self,
        x0=None,
        sigma=0.1,
        opts=None,
        normalize=True,
        repair_bounds=True,
        **kwargs
    ):
        super().__init__(x0=x0, **kwargs)

        self.termination = NoTermination()
        self.es = None
        self.sigma = float(sigma)

        self.normalize = normalize
        self.repair_bounds = repair_bounds

        self.norm = None
        self.xl = None
        self.xu = None
        self.internal_lb = None
        self.internal_ub = None

        if opts is None:
            opts = {}

        self.opts = dict(opts)

        self.opts.setdefault("verb_disp", 0)
        self.opts.setdefault("verbose", -9)
        self.opts.setdefault("verb_log", 0)

        self.opts = self._clean_opts(self.opts)

    def _clean_opts(self, opts):
        """
        Limpia opciones que activan el manejo interno de bounds de pycma.
        Los bounds los manejamos manualmente con np.clip.
        """
        clean_opts = dict(opts)

        for key in [
            "bounds",
            "BoundaryHandler",
            "boundary_handler",
            "transformation",
            "fixed_variables"
        ]:
            clean_opts.pop(key, None)

        return clean_opts

    def _setup(self, problem, **kwargs):
        xl, xu = problem.bounds()

        self.xl = np.asarray(xl, dtype=float)
        self.xu = np.asarray(xu, dtype=float)

        if self.normalize:
            self.norm = ZeroToOneNormalization(xl=self.xl, xu=self.xu)
            self.internal_lb = np.zeros_like(self.xl, dtype=float)
            self.internal_ub = np.ones_like(self.xu, dtype=float)
        else:
            self.norm = NoNormalization()
            self.internal_lb = self.xl.copy()
            self.internal_ub = self.xu.copy()

        self.opts = self._clean_opts(self.opts)

        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            self.opts["seed"] = int(seed)

        if isinstance(self.termination, MaximumGenerationTermination):
            self.opts["maxiter"] = int(self.termination.n_max_gen)

        elif isinstance(self.termination, MaximumFunctionCallTermination):
            self.opts["maxfevals"] = int(self.termination.n_max_evals)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)

        x = np.asarray(self.norm.forward(self.x0.X), dtype=float).reshape(-1)

        if self.repair_bounds:
            x = np.clip(x, self.internal_lb, self.internal_ub)

        if x.size != self.internal_lb.size:
            raise RuntimeError(
                f"x0 tiene dimensión incorrecta. "
                f"x0.size={x.size}, pero el problema espera {self.internal_lb.size} variables."
            )

        clean_opts = self._clean_opts(self.opts)
 
        self.es = cma.CMAEvolutionStrategy(
            x.tolist(),
            self.sigma,
            clean_opts
        )

    def _infill(self):
        X_internal = np.asarray(self.es.ask(), dtype=float)
        X_internal = np.atleast_2d(X_internal)

        if self.repair_bounds:
            X_internal = np.clip(
                X_internal,
                self.internal_lb,
                self.internal_ub
            )

        X_real = np.asarray(self.norm.backward(X_internal), dtype=float)
        X_real = np.clip(X_real, self.xl, self.xu)

        return Population.new("X", X_real)

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            self.termination.force_termination = True
            return

        X, F = infills.get("X", "F")

        X_internal = np.asarray(self.norm.forward(X), dtype=float)

        if self.repair_bounds:
            X_internal = np.clip(
                X_internal,
                self.internal_lb,
                self.internal_ub
            )

        F = np.asarray(F, dtype=float).reshape(-1)
        F = np.where(np.isfinite(F), F, np.inf)

        self.es.tell(
                    np.asarray(X_internal, dtype=float),
                    np.asarray(F, dtype=float)
                )

        self.pop = infills

        if self.es.stop():
            self.termination.force_termination = True

    def _set_optimum(self):
        pop = self.pop if self.opt is None else Population.merge(self.opt, self.pop)
        self.opt = filter_optimum(pop, least_infeasible=True)