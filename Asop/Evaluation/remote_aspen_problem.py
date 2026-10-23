import json
import math
import sys
import uuid

import numpy as np
from pymoo.core.problem import Problem


PENALTY_VALUE = 1e10
PROTOCOL_PREFIX = "__MPL__"


class CSharpEvaluationClient:
    """Send Aspen evaluation requests to the C# JSONL bridge.

    Python proposes points and computes objective expressions, but Aspen COM is
    owned exclusively by the C# process.
    """

    def evaluate_batch(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        request_id = uuid.uuid4().hex
        request = {
            "type": "evaluate_batch",
            "request_id": request_id,
            "X": X.tolist(),
        }

        sys.stdout.write(PROTOCOL_PREFIX + json.dumps(request, separators=(",", ":")) + "\n")
        sys.stdout.flush()

        line = sys.stdin.readline()
        if line == "":
            raise RuntimeError(
                "C# closed the JSONL bridge while Python was waiting for an Aspen batch result."
            )

        response = json.loads(line)

        if response.get("type") != "batch_result":
            raise RuntimeError(f"Unexpected C# response type: {response.get('type')}")

        if response.get("request_id") != request_id:
            raise RuntimeError(
                "C# batch response does not match the current request. "
                f"Expected {request_id}, received {response.get('request_id')}."
            )

        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("C# batch response does not contain a results list.")

        if len(results) != len(X):
            raise RuntimeError(
                f"C# returned {len(results)} evaluations for a batch containing {len(X)} points."
            )

        return results


class CSharpAspenEvaluator:
    """Reusable Aspen evaluator for pymoo, SBOA and Bayesian optimizers.

    Input-only hard constraints and objective expressions remain in Python.
    Every physical simulator evaluation is delegated to C# through
    ``CSharpEvaluationClient``.
    """

    def __init__(self, config):
        self.config = config
        self.client = CSharpEvaluationClient()
        self.objectives = config.get("objectives", []) or []
        self.objective_names = [
            self.get_objective_name(obj, i)
            for i, obj in enumerate(self.objectives)
        ]

        if len(self.objectives) == 0:
            raise ValueError("At least one objective is required.")

    def preprocess_x(self, x):
        x_processed = []
        for input_var, value in zip(self.config.get("inputs", []), x):
            variable_type = str(input_var.get("variable_type", "Continuous")).strip().lower()
            if variable_type in {"discrete", "integer"}:
                value = round(float(value))
            x_processed.append(float(value))
        return x_processed

    def input_context(self, x):
        ctx = {}
        for input_var, value in zip(self.config.get("inputs", []), x):
            tag = input_var.get("tag") or input_var.get("alias")
            if tag:
                ctx[tag] = float(value)
        return ctx

    @staticmethod
    def normalize_key(d, *names, default=None):
        if not isinstance(d, dict):
            return default
        lower_map = {str(k).lower().replace("_", ""): k for k in d.keys()}
        for name in names:
            key = lower_map.get(str(name).lower().replace("_", ""))
            if key is not None:
                return d[key]
        return default

    @staticmethod
    def resolve_value(token, ctx):
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

    @staticmethod
    def compare_values(left, operator, right, tol=1e-9):
        op = str(operator).strip().replace("≤", "<=").replace("≥", ">=").replace("==", "=")
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
        raise ValueError(f"Unsupported constraint operator: {operator}")

    def check_input_constraints(self, x):
        constraints = self.config.get("constraints", []) or []
        ctx = self.input_context(x)
        violations = []

        for i, constraint in enumerate(constraints, start=1):
            ctype = str(
                self.normalize_key(constraint, "type", "constraint_type", default="Hard")
            ).lower()
            if ctype == "soft":
                continue

            left_token = self.normalize_key(
                constraint, "left_side", "leftside", "left", default=None
            )
            operator = self.normalize_key(constraint, "operator", "op", default=None)
            right_token = self.normalize_key(
                constraint, "right_side", "rightside", "right", default=None
            )

            if left_token in [None, ""] or operator in [None, ""] or right_token in [None, ""]:
                continue

            try:
                left_value = self.resolve_value(left_token, ctx)
                right_value = self.resolve_value(right_token, ctx)
                ok = self.compare_values(left_value, operator, right_value)
            except Exception as exc:
                violations.append(
                    {
                        "constraint": i,
                        "message": f"Constraint could not be evaluated: {exc}",
                    }
                )
                continue

            if not ok:
                violations.append(
                    {
                        "constraint": i,
                        "message": f"Constraint not satisfied: {left_token} {operator} {right_token}",
                    }
                )

        return len(violations) == 0, violations

    @staticmethod
    def get_objective_name(obj, i):
        return str(
            obj.get("tag")
            or obj.get("alias")
            or obj.get("name")
            or obj.get("objective_name")
            or f"f{i + 1}"
        )

    @staticmethod
    def get_objective_sense(obj):
        sense = str(
            obj.get("sense")
            or obj.get("Sense")
            or obj.get("type")
            or obj.get("Type")
            or "minimize"
        ).strip().lower()

        if sense in {"minimize", "min", "minimum", "minimizar"}:
            return "minimize"
        if sense in {"maximize", "max", "maximum", "maximizar"}:
            return "maximize"
        raise ValueError(f"Unsupported objective sense: {sense}")

    def evaluate_objectives(self, x, outputs):
        input_ctx = self.input_context(x)
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

        minimized = []
        raw = []

        for i, obj in enumerate(self.objectives):
            expression = obj.get("expression") or obj.get("Expression")
            if expression is None or str(expression).strip() == "":
                raise RuntimeError(f"Objective {i + 1} has no expression.")

            raw_value = float(eval(expression, safe_globals, safe_locals))
            if not np.isfinite(raw_value):
                raise RuntimeError(
                    f"Objective {self.objective_names[i]} is not finite: {raw_value}"
                )

            minimized_value = (
                -raw_value if self.get_objective_sense(obj) == "maximize" else raw_value
            )
            raw.append(raw_value)
            minimized.append(minimized_value)

        return minimized, raw

    def restore_raw_objectives(self, F):
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

    def penalty_result(self, x, reason, run_status=None):
        outputs = {
            output_var["tag"]: None
            for output_var in self.config.get("outputs", [])
        }
        return {
            "objective": float(PENALTY_VALUE),
            "objective_values": [float(PENALTY_VALUE)] * len(self.objectives),
            "objective_values_raw": [None] * len(self.objectives),
            "outputs": outputs,
            "x": list(x),
            "feasible": False,
            "aspen_converged": False,
            "run_status": run_status,
            "message": reason,
        }

    def evaluate_batch(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        processed = [self.preprocess_x(row) for row in X]
        row_results = [None] * len(processed)
        remote_indices = []
        remote_points = []

        # Reject input-only hard-constraint violations before consuming Aspen time.
        for i, x in enumerate(processed):
            ok, _violations = self.check_input_constraints(x)
            if not ok:
                row_results[i] = self.penalty_result(
                    x,
                    "Point rejected before Aspen because an input constraint was not satisfied.",
                )
            else:
                remote_indices.append(i)
                remote_points.append(x)

        if remote_points:
            remote_results = self.client.evaluate_batch(remote_points)

            for source_index, physical in zip(remote_indices, remote_results):
                x = processed[source_index]
                outputs = physical.get("outputs") or {}

                if not bool(physical.get("success", False)):
                    row_results[source_index] = self.penalty_result(
                        x,
                        physical.get("message", "Aspen evaluation failed in C#."),
                        run_status=physical.get("run_status"),
                    )
                    continue

                try:
                    objective_values, objective_values_raw = self.evaluate_objectives(x, outputs)
                    row_results[source_index] = {
                        "objective": float(objective_values[0]),
                        "objective_values": [float(v) for v in objective_values],
                        "objective_values_raw": [float(v) for v in objective_values_raw],
                        "outputs": outputs,
                        "x": list(x),
                        "feasible": True,
                        "aspen_converged": bool(physical.get("aspen_converged", True)),
                        "run_status": physical.get("run_status"),
                        "message": physical.get("message", ""),
                    }
                except Exception as exc:
                    row_results[source_index] = self.penalty_result(
                        x,
                        f"Error evaluating objective expression in Python: {exc}",
                        run_status=physical.get("run_status"),
                    )

        if any(result is None for result in row_results):
            raise RuntimeError("Internal error: an Aspen evaluation result was not assigned.")

        return row_results

    def evaluate_one(self, x):
        return self.evaluate_batch([x])[0]


class _RemoteAspenProblemBase(Problem):
    """Vectorized pymoo wrapper around ``CSharpAspenEvaluator``."""

    def __init__(self, config, n_obj):
        self.config = config
        self.history = []
        self.evaluator = CSharpAspenEvaluator(config)
        self.objectives = self.evaluator.objectives
        self.objective_names = self.evaluator.objective_names

        if len(self.objectives) != n_obj:
            raise ValueError(
                f"Expected {n_obj} objective(s), but config contains {len(self.objectives)}."
            )

        xl = np.array([float(v["lower_bound"]) for v in config["inputs"]], dtype=float)
        xu = np.array([float(v["upper_bound"]) for v in config["inputs"]], dtype=float)

        super().__init__(
            n_var=len(config["inputs"]),
            n_obj=n_obj,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
        )

    def restore_raw_objectives(self, F):
        return self.evaluator.restore_raw_objectives(F)

    def _evaluate_population(self, X):
        results = self.evaluator.evaluate_batch(X)

        for result in results:
            objective_values = list(result["objective_values"])
            record = {
                "evaluation": len(self.history) + 1,
                "x": {
                    self.config["inputs"][i]["tag"]: float(result["x"][i])
                    for i in range(len(result["x"]))
                },
                "objective": float(objective_values[0]),
                "objective_values": [float(v) for v in objective_values],
                "objective_values_raw": result.get("objective_values_raw"),
                "outputs": result["outputs"],
                "feasible": bool(result.get("feasible", False)),
                "aspen_converged": bool(result.get("aspen_converged", False)),
                "run_status": result.get("run_status"),
                "message": result.get("message", ""),
            }
            self.history.append(record)

        return results


class RemoteAspenProblem(_RemoteAspenProblemBase):
    """Vectorized single-objective problem evaluated physically by C# Aspen."""

    def __init__(self, config):
        if len(config.get("objectives", []) or []) != 1:
            raise ValueError("RemoteAspenProblem requires exactly one objective.")
        super().__init__(config, n_obj=1)

    def _evaluate(self, X, out, *args, **kwargs):
        results = self._evaluate_population(X)
        out["F"] = np.asarray(
            [[float(result["objective_values"][0])] for result in results],
            dtype=float,
        )


class RemoteAspenProblemMOBO(_RemoteAspenProblemBase):
    """Vectorized multi-objective problem evaluated physically by C# Aspen."""

    def __init__(self, config):
        n_obj = len(config.get("objectives", []) or [])
        if n_obj < 2:
            raise ValueError("RemoteAspenProblemMOBO requires at least two objectives.")
        super().__init__(config, n_obj=n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        results = self._evaluate_population(X)
        out["F"] = np.asarray(
            [result["objective_values"] for result in results],
            dtype=float,
        )
