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
from dataclasses import dataclass
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf

import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

from bayesian_classes import VanillaBOBotorch 

RUN_STATUS_DIR = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
FAILED_RUN_STATUS_VALUES = {9, 10}
PENALTY_VALUE = 1e10

try:
    from torch.quasirandom import SobolEngine
except Exception:
    SobolEngine = None

try:
    from botorch.generation import MaxPosteriorSampling
except Exception:
    MaxPosteriorSampling = None

try:
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
except Exception:
    try:
        from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
    except Exception:
        qExpectedHypervolumeImprovement = None

try:
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
except Exception:
    try:
        from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
    except Exception:
        NondominatedPartitioning = None


# =========================================================
# Trust Region Bayesian Optimization (continuous variables)
# Adaptado para Aspen desde bo(1).py, sin BioSTEAM ni VAE.
# =========================================================

@dataclass
class AspenTurboState:
    dim: int
    batch_size: int = 1
    length: float = 0.5
    length_min: float = 0.5**30
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = -1
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    shrink_factor: float = 0.7
    expand_factor: float = 2.0
    abs_tol: float = 1e-6
    rel_tol: float = 1e-3

    def __post_init__(self):
        if self.failure_tolerance is None or int(self.failure_tolerance) <= 0:
            self.failure_tolerance = math.ceil(max(4.0 / self.batch_size, float(self.dim) / self.batch_size))


def _only_continuous_config(config):
    return all(str(v.get("variable_type", "Continuous")).strip().lower() == "continuous"
               for v in config.get("inputs", []))


class TrustRegionBOBotorch(VanillaBOBotorch):
    """TuRBO-style single-objective BO for continuous Aspen variables.

    The internal BoTorch model maximizes -objective because the rest of the
    application stores single-objective values in minimization form.
    """

    def __init__(self, config, aspen):
        super().__init__(config, aspen)
        if not _only_continuous_config(config):
            raise RuntimeError(
                "Trust Region Bayesian Optimization solo está habilitado para variables continuas. "
                "Cambia todas las variables de entrada a Continuous."
            )

        self.acqf_raw = str(self.hyper.get("acqf", self.hyper.get("aqf", "Log-EI"))).strip()
        self.acqf = self._normalize_acquisition_name(self.acqf_raw)
        print(f"[TuRBO] Acquisition selected: raw='{self.acqf_raw}' normalized='{self.acqf}'", flush=True)

        self.batch_size = int(float(self.hyper.get("batch_size", 1)))
        self.n_init = int(float(self.hyper.get("n_init", max(2 * self.dim, 5))))
        self.n_iter = int(float(self.hyper.get("n_iter", 30)))
        self.n_candidates = int(float(self.hyper.get("n_candidates", max(2000, 200 * self.dim))))
        self.num_restarts = int(float(self.hyper.get("num_restarts", 10)))
        self.raw_samples = int(float(self.hyper.get("raw_samples", 512)))

        self.length = float(self.hyper.get("length", 0.5))
        self.length_min = float(self.hyper.get("length_min", 0.5**30))
        self.length_max = float(self.hyper.get("length_max", 1.6))
        self.success_tolerance = int(float(self.hyper.get("success_tolerance", 3)))
        self.failure_tolerance = int(float(self.hyper.get("failure_tolerance", -1)))
        self.shrink_factor = float(self.hyper.get("shrink_factor", 0.7))
        self.expand_factor = float(self.hyper.get("expand_factor", 2.0))
        self.abs_tol = float(self.hyper.get("abs_tol", 1e-6))
        self.rel_tol = float(self.hyper.get("rel_tol", 1e-3))
        self.beta = float(self.hyper.get("beta", 0.5))

        if np.any(self.ub <= self.lb):
            raise RuntimeError("Error en bounds: todos los upper_bound deben ser mayores que lower_bound.")

    def _normalize_acquisition_name(self, name):
        text = str(name or "").strip().lower()
        text = text.replace("_", "-").replace(" ", "-")

        if text in ["log-ei", "logei", "ei", "expected-improvement", "expectedimprovement"]:
            return "logei"

        if text in ["upper-confidence-bound", "ucb"]:
            return "ucb"

        if text in ["thompson-sampling", "thomson-sampling", "thompson", "thomson", "ts"]:
            return "ts"

        raise RuntimeError(
            "Acquisition function no reconocida para TuRBO: "
            f"'{name}'. Usa: Log-EI, Upper Confidence Bound o Thomson/Thompson Sampling."
        )

    def evaluate_and_store(self, x):
        record = super().evaluate_and_store(x)
        record["algorithm"] = "Trust Region Bayesian Optimization"
        record["acquisition_function_raw"] = self.acqf_raw
        record["acquisition_function"] = self.acqf
        return record

    def _sobol_points(self, n, seed=None):
        """
        Genera puntos Sobol en [0, 1]^d.

        Nota:
        - Para el diseño inicial usamos self.seed para que Log-EI y UCB sean comparables.
        - Para la optimización de la adquisición usamos una semilla dependiente de
          la adquisición y de la iteración, evitando que todas las adquisiciones
          arranquen con exactamente los mismos raw samples.
        """
        seed_to_use = self.seed if seed is None else int(seed)

        if SobolEngine is not None:
            sobol = SobolEngine(self.dim, scramble=True, seed=seed_to_use)
            return sobol.draw(n).to(dtype=torch.double)

        rng = np.random.default_rng(seed_to_use)
        return torch.tensor(rng.random((n, self.dim)), dtype=torch.double)

    def sample_random_point(self, observed_keys=None, tr_lb=None, tr_ub=None, max_tries=5000):
        observed_keys = observed_keys or set()
        for _ in range(max_tries):
            xn = self.rng.random(self.dim)
            if tr_lb is not None and tr_ub is not None:
                lo = tr_lb.detach().cpu().numpy().reshape(-1)
                hi = tr_ub.detach().cpu().numpy().reshape(-1)
                xn = lo + (hi - lo) * xn
            x = self.round_to_valid_x(self.denormalize_x(xn))
            key = self.point_key(x)
            if key in observed_keys:
                continue
            if not self.satisfies_input_constraints(x):
                continue
            return x
        return None

    def initial_design(self, observed_keys):
        X01 = self._sobol_points(max(self.n_init * 3, self.n_init)).cpu().numpy()
        for xn in X01:
            x = self.round_to_valid_x(self.denormalize_x(xn))
            key = self.point_key(x)
            if key in observed_keys:
                continue
            if not self.satisfies_input_constraints(x):
                continue
            observed_keys.add(key)
            record = self.evaluate_and_store(x)
            record["turbo_iteration"] = 0
            record["candidate_source"] = "initial_sobol"
            if len(self.history_records) >= self.n_init:
                break

        while len(self.history_records) < self.n_init:
            x = self.sample_random_point(observed_keys)
            if x is None:
                break
            observed_keys.add(self.point_key(x))
            record = self.evaluate_and_store(x)
            record["turbo_iteration"] = 0
            record["candidate_source"] = "initial_random"

    def get_valid_training_data(self):
        valid_records = [
            r for r in self.history_records
            if r.get("feasible") and r.get("aspen_converged") and np.isfinite(r.get("objective", np.nan))
        ]
        if len(valid_records) == 0:
            return None, None, []

        X_np = np.array([
            [r["x"][v["tag"]] for v in self.config["inputs"]]
            for r in valid_records
        ], dtype=float)
        Y_np = np.array([[-float(r["objective"])] for r in valid_records], dtype=float)

        train_X = torch.tensor(np.array([self.normalize_x(x) for x in X_np]), dtype=torch.double)
        train_Y = torch.tensor(Y_np, dtype=torch.double)
        return train_X, train_Y, valid_records

    def _trust_region_bounds(self, model, x_center, state):
        try:
            lengthscale = model.covar_module.base_kernel.lengthscale.detach().view(-1).to(dtype=torch.double)
            weights = lengthscale / lengthscale.mean().clamp_min(1e-12)
            weights = weights / torch.prod(weights.pow(1.0 / len(weights))).clamp_min(1e-12)
        except Exception:
            weights = torch.ones(self.dim, dtype=torch.double)

        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
        return tr_lb, tr_ub

    def _acquisition_seed(self, iteration):
        offsets = {
            "logei": 0,
            "ucb": 104729,
            "ts": 209759,
        }
        return int(self.seed + offsets.get(self.acqf, 0) + int(iteration))

    def _select_from_candidate_set(self, acquisition, tr_lb, tr_ub, iteration, source_label):
        """
        Optimización discreta de la adquisición dentro de la trust region.

        Para simulaciones caras tipo Aspen, esto suele ser más estable y más
        transparente que optimize_acqf porque evalúa una nube Sobol grande y
        escoge el punto con mayor adquisición. Además, permite que Log-EI y UCB
        tengan semillas de candidatos distintas.
        """
        acq_seed = self._acquisition_seed(iteration)
        X_cand = self._sobol_points(self.n_candidates, seed=acq_seed)
        X_cand = tr_lb + (tr_ub - tr_lb) * X_cand

        with torch.no_grad():
            acq_values = acquisition(X_cand.unsqueeze(-2)).reshape(-1)

        best_ids = torch.argsort(acq_values, descending=True)[:self.batch_size]
        X_next = X_cand[best_ids].detach()

        self.last_candidate_source = source_label
        self.last_acq_value = float(acq_values[best_ids[0]].item())
        self.last_tr_lb = tr_lb.detach().cpu().numpy().reshape(-1).tolist()
        self.last_tr_ub = tr_ub.detach().cpu().numpy().reshape(-1).tolist()

        return X_next, tr_lb, tr_ub

    def generate_batch(self, state, model, train_X, train_Y, iteration=0):
        best_idx = int(torch.argmax(train_Y).item())
        x_center = train_X[best_idx].clone().detach()
        tr_lb, tr_ub = self._trust_region_bounds(model, x_center, state)
        acqf = self.acqf

        self.last_candidate_source = None
        self.last_acq_value = None
        self.last_tr_lb = tr_lb.detach().cpu().numpy().reshape(-1).tolist()
        self.last_tr_ub = tr_ub.detach().cpu().numpy().reshape(-1).tolist()

        if acqf == "ts" and MaxPosteriorSampling is not None:
            acq_seed = self._acquisition_seed(iteration)
            X_cand = self._sobol_points(self.n_candidates, seed=acq_seed)
            X_cand = tr_lb + (tr_ub - tr_lb) * X_cand

            prob_perturb = min(20.0 / self.dim, 1.0) * float(state.length / state.length_max)
            prob_perturb = max(prob_perturb, 0.2)

            with torch.random.fork_rng():
                torch.manual_seed(acq_seed)
                mask = torch.rand(self.n_candidates, self.dim, dtype=torch.double) <= prob_perturb
                empty = torch.where(mask.sum(dim=1) == 0)[0]
                if len(empty) > 0:
                    mask[empty, torch.randint(0, self.dim, size=(len(empty),))] = True

            X_centered = x_center.expand(self.n_candidates, self.dim).clone()
            X_centered[mask] = X_cand[mask]
            sampler = MaxPosteriorSampling(model=model, replacement=False)

            with torch.no_grad():
                X_next = sampler(X_centered, num_samples=self.batch_size)

            self.last_candidate_source = "thompson_sampling_candidate_set"
            return X_next.detach(), tr_lb, tr_ub

        if acqf == "logei":
            acquisition = LogExpectedImprovement(model, best_f=train_Y.max().item())
            return self._select_from_candidate_set(
                acquisition=acquisition,
                tr_lb=tr_lb,
                tr_ub=tr_ub,
                iteration=iteration,
                source_label="logei_sobol_candidate_set",
            )

        if acqf == "ucb":
            acquisition = UpperConfidenceBound(model, beta=self.beta)
            return self._select_from_candidate_set(
                acquisition=acquisition,
                tr_lb=tr_lb,
                tr_ub=tr_ub,
                iteration=iteration,
                source_label=f"ucb_sobol_candidate_set_beta_{self.beta}",
            )

        raise RuntimeError(f"Acquisition function no soportada en generate_batch: {acqf}")

    def update_state(self, state, y_next):
        thresh = max(state.abs_tol, state.rel_tol * max(1.0, abs(state.best_value)))
        y_best_new = float(y_next.max().item())

        if y_best_new > state.best_value + thresh:
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter >= state.success_tolerance:
            state.length = min(state.expand_factor * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter >= state.failure_tolerance:
            state.length *= state.shrink_factor
            state.failure_counter = 0

        state.best_value = max(state.best_value, y_best_new)
        if state.length < state.length_min:
            state.restart_triggered = True
        return state

    def run(self):
        observed_keys = set()
        self.initial_design(observed_keys)

        train_X, train_Y, valid_records = self.get_valid_training_data()
        best_value = float(train_Y.max().item()) if train_Y is not None and len(train_Y) else -float("inf")
        state = AspenTurboState(
            dim=self.dim,
            batch_size=self.batch_size,
            length=self.length,
            length_min=self.length_min,
            length_max=self.length_max,
            failure_tolerance=self.failure_tolerance,
            success_tolerance=self.success_tolerance,
            best_value=best_value,
            shrink_factor=self.shrink_factor,
            expand_factor=self.expand_factor,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
        )

        for iteration in range(self.n_iter):
            train_X, train_Y, valid_records = self.get_valid_training_data()
            if train_X is None or len(valid_records) < 2:
                x_next = self.sample_random_point(observed_keys)
                if x_next is None:
                    break
                observed_keys.add(self.point_key(x_next))
                record = self.evaluate_and_store(x_next)
                record["turbo_iteration"] = int(iteration + 1)
                record["trust_region_length"] = float(state.length)
                record["candidate_source"] = "fallback_random"
                if record.get("feasible") and np.isfinite(record.get("objective", np.nan)):
                    y_next = torch.tensor([[-float(record["objective"])]], dtype=torch.double)
                    state = self.update_state(state, y_next)
                continue

            model = self.build_model(train_X, train_Y)

            try:
                X_next_norm, tr_lb, tr_ub = self.generate_batch(state, model, train_X, train_Y, iteration=iteration)
                X_next_np = X_next_norm.detach().cpu().numpy().reshape(-1, self.dim)
            except Exception as e:
                raise RuntimeError(
                    f"Error en TuRBO acquisition. raw='{self.acqf_raw}', normalized='{self.acqf}'. "
                    f"Detalle: {e}"
                ) from e

            evaluated_any = False
            y_values = []
            for xn in X_next_np:
                x_candidate = self.round_to_valid_x(self.denormalize_x(xn))
                key = self.point_key(x_candidate)
                if key in observed_keys or not self.satisfies_input_constraints(x_candidate):
                    x_candidate = self.sample_random_point(observed_keys, tr_lb, tr_ub)
                    if x_candidate is None:
                        x_candidate = self.sample_random_point(observed_keys)
                if x_candidate is None:
                    continue

                key = self.point_key(x_candidate)
                observed_keys.add(key)
                record = self.evaluate_and_store(x_candidate)
                record["turbo_iteration"] = int(iteration + 1)
                record["trust_region_length"] = float(state.length)
                record["candidate_source"] = getattr(self, "last_candidate_source", None)
                record["acq_value"] = getattr(self, "last_acq_value", None)
                record["tr_lb"] = getattr(self, "last_tr_lb", None)
                record["tr_ub"] = getattr(self, "last_tr_ub", None)

                evaluated_any = True
                if record.get("feasible") and record.get("aspen_converged") and np.isfinite(record.get("objective", np.nan)):
                    y_values.append(-float(record["objective"]))
                else:
                    y_values.append(-PENALTY_VALUE)

            if evaluated_any and len(y_values) > 0:
                state = self.update_state(state, torch.tensor(y_values, dtype=torch.double).reshape(-1, 1))
            else:
                state.failure_counter += 1
                if state.failure_counter >= state.failure_tolerance:
                    state.length *= state.shrink_factor
                    state.failure_counter = 0

            if state.restart_triggered:
                break

        valid_records = [
            row for row in self.history_records
            if row.get("feasible") and row.get("aspen_converged") and np.isfinite(row.get("objective", np.nan))
        ]

        if len(valid_records) == 0:
            return {
                "status": "error",
                "message": "Trust Region Bayesian Optimization terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "best_x": None,
                "best_objective": None,
                "outputs": None,
                "evaluations": self.history_records,
                "trust_region_length": state.length,
                "acquisition_function_raw": self.acqf_raw,
                "acquisition_function": self.acqf,
            }, self.history_records

        best_record = min(valid_records, key=lambda r: r["objective"])
        return {
            "status": "success",
            "message": f"Trust Region Bayesian Optimization finished using {self.acqf_raw} ({self.acqf}).",
            "best_x": best_record["x"],
            "best_objective": float(best_record["objective"]),
            "outputs": best_record["outputs"],
            "evaluations": self.history_records,
            "trust_region_length": state.length,
            "restart_triggered": bool(state.restart_triggered),
            "acquisition_function_raw": self.acqf_raw,
            "acquisition_function": self.acqf,
        }, self.history_records


# =========================================================
# Continuous Multi-Objective Bayesian Optimization
# Inspired by BO/VAE workflow, but operating directly in continuous x-space.
# =========================================================

class ContinuousMOBOBotorch(VanillaBOBotorch):
    """Multi-objective BO for continuous Aspen variables.

    Stores F in minimization convention, and uses Y=-F for BoTorch EHVI because
    BoTorch acquisitions maximize by default.
    """

    def __init__(self, config, aspen):
        super().__init__(config, aspen)
        if not _only_continuous_config(config):
            raise RuntimeError(
                "Vanilla M.O. Bayesian Optimization solo está habilitado para variables continuas. "
                "Cambia todas las variables de entrada a Continuous."
            )

        self.objectives = config.get("objectives", []) or []
        if len(self.objectives) < 2:
            raise RuntimeError("Multi-objective Bayesian Optimization requiere al menos dos funciones objetivo.")

        self.objective_names = [
            str(obj.get("name") or obj.get("tag") or obj.get("alias") or f"obj{i + 1}")
            for i, obj in enumerate(self.objectives)
        ]
        self.n_obj = len(self.objectives)
        self.batch_size = int(float(self.hyper.get("batch_size", 1)))
        self.n_init = int(float(self.hyper.get("n_init", max(2 * self.dim + 2, 2 * self.n_obj + 4))))
        self.n_iter = int(float(self.hyper.get("n_iter", 30)))
        self.num_restarts = int(float(self.hyper.get("num_restarts", 10)))
        self.raw_samples = int(float(self.hyper.get("raw_samples", 512)))
        self.ref_point_raw = self.hyper.get("ref_point", self.hyper.get("reference_point", "auto"))

    def evaluate_point_multi(self, aspen, config, x):
        x = self.preprocess_x(config, x)
        ok_constraints, violations = self.check_input_constraints(config, x)
        if not ok_constraints:
            return self.penalty_result_multi(
                config=config,
                x=x,
                reason="Punto rechazado antes de correr Aspen porque no cumple restricciones de entrada.",
                violations=violations,
            )

        try:
            self.update_aspen(aspen, config)
            for input_var, value in zip(config["inputs"], x):
                self.set_aspen_variable(aspen, input_var["path"], value)
            self.run_aspen(aspen)

            converged, run_status, status_message = self.check_aspen_convergence(aspen)
            if not converged:
                self.reinit_aspen(aspen)
                return self.penalty_result_multi(config, x, status_message, run_status=run_status)

            outputs = {}
            for output_var in config["outputs"]:
                outputs[output_var["tag"]] = self.get_aspen_variable(aspen, output_var["path"])

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

            f_minimized = []
            f_raw = []
            for obj in self.objectives:
                value_raw = float(eval(obj["expression"], safe_globals, safe_locals))
                if not np.isfinite(value_raw):
                    raise RuntimeError(f"La función objetivo no es finita: {value_raw}")
                sense = str(obj.get("sense", "Minimize")).lower()
                value_min = -value_raw if sense == "maximize" else value_raw
                f_raw.append(value_raw)
                f_minimized.append(value_min)

            return {
                "objective_values": f_minimized,
                "objective_values_raw": f_raw,
                "outputs": outputs,
                "x": x,
                "feasible": True,
                "aspen_converged": True,
                "run_status": run_status,
                "message": status_message,
            }

        except Exception as e:
            self.reinit_aspen(aspen)
            return self.penalty_result_multi(config, x, f"Error al evaluar el punto en Aspen: {e}")

    def penalty_result_multi(self, config, x, reason, violations=None, run_status=None):
        x = self.preprocess_x(config, x)
        outputs = {output_var["tag"]: None for output_var in config.get("outputs", [])}
        return {
            "objective_values": [PENALTY_VALUE for _ in self.objectives],
            "objective_values_raw": [None for _ in self.objectives],
            "outputs": outputs,
            "x": x,
            "feasible": False,
            "aspen_converged": False,
            "run_status": run_status,
            "message": reason,
        }

    def evaluate_and_store_multi(self, x):
        x = self.round_to_valid_x(x)
        result = self.evaluate_point_multi(self.aspen, self.config, x)
        record = {
            "evaluation": len(self.history_records) + 1,
            "x": {
                self.config["inputs"][i]["tag"]: float(result["x"][i])
                for i in range(len(result["x"]))
            },
            "objective_values": [float(v) for v in result["objective_values"]],
            "objective_values_raw": result.get("objective_values_raw"),
            "outputs": result["outputs"],
            "feasible": bool(result["feasible"]),
            "aspen_converged": bool(result["aspen_converged"]),
            "run_status": result["run_status"],
            "message": result["message"],
        }
        self.history_records.append(record)
        return record

    def initial_design_multi(self, observed_keys):
        if SobolEngine is not None:
            sobol = SobolEngine(self.dim, scramble=True, seed=self.seed)
            X01 = sobol.draw(max(self.n_init * 3, self.n_init)).cpu().numpy()
        else:
            X01 = self.rng.random((max(self.n_init * 3, self.n_init), self.dim))

        for xn in X01:
            x = self.round_to_valid_x(self.denormalize_x(xn))
            key = self.point_key(x)
            if key in observed_keys or not self.satisfies_input_constraints(x):
                continue
            observed_keys.add(key)
            self.evaluate_and_store_multi(x)
            if len(self.history_records) >= self.n_init:
                break

        while len(self.history_records) < self.n_init:
            x = self.random_unobserved(observed_keys)
            if x is None:
                break
            observed_keys.add(self.point_key(x))
            self.evaluate_and_store_multi(x)

    def get_valid_multi_records(self):
        return [
            r for r in self.history_records
            if r.get("feasible") and r.get("aspen_converged")
            and all(np.isfinite(v) for v in r.get("objective_values", []))
        ]

    def build_training_data_multi(self, valid_records):
        X_np = np.array([
            [r["x"][v["tag"]] for v in self.config["inputs"]]
            for r in valid_records
        ], dtype=float)
        F_np = np.array([r["objective_values"] for r in valid_records], dtype=float)
        train_X = torch.tensor(np.array([self.normalize_x(x) for x in X_np]), dtype=torch.double)
        train_Y = torch.tensor(-F_np, dtype=torch.double)  # BoTorch maximization convention
        return train_X, train_Y, F_np

    def build_model_list(self, train_X, train_Y):
        models = []
        for i in range(train_Y.shape[1]):
            model_i = SingleTaskGP(
                train_X,
                train_Y[:, i:i+1],
                outcome_transform=Standardize(m=1),
                likelihood=GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-1)),
                covar_module=ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=self.dim,
                        lengthscale_constraint=Interval(0.01, 10.0),
                    )
                ),
            )
            models.append(model_i)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def parse_ref_point_y(self, train_Y):
        raw = str(self.ref_point_raw).strip().lower()
        if raw in ["", "auto", "none"]:
            y_min = train_Y.min(dim=0).values
            y_max = train_Y.max(dim=0).values
            margin = 0.1 * torch.clamp(y_max - y_min, min=1e-6)
            return (y_min - margin).to(dtype=torch.double)

        try:
            values = [float(v.strip()) for v in str(self.ref_point_raw).replace(";", ",").split(",") if v.strip()]
            if len(values) != self.n_obj:
                raise ValueError
            # El ref_point se interpreta en espacio Y=maximización para evitar ambigüedad con objetivos Maximize.
            return torch.tensor(values, dtype=torch.double)
        except Exception:
            raise RuntimeError(
                f"ref_point debe ser 'auto' o una lista de {self.n_obj} valores en espacio de maximización Y=-F."
            )

    def propose_mobo_candidate(self, train_X, train_Y, observed_keys):
        bounds = torch.stack([
            torch.zeros(self.dim, dtype=torch.double),
            torch.ones(self.dim, dtype=torch.double),
        ])

        if qExpectedHypervolumeImprovement is not None and NondominatedPartitioning is not None:
            model = self.build_model_list(train_X, train_Y)
            ref_point = self.parse_ref_point_y(train_Y)
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
            acquisition = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                partitioning=partitioning,
            )
            X_next, _ = optimize_acqf(  acquisition,
                                        bounds=bounds,
                                        q=self.batch_size,
                                        num_restarts=self.num_restarts,
                                        raw_samples=self.raw_samples,
                                        options={"batch_limit": 5, "maxiter": 200},
                                    )
            return X_next.detach().cpu().numpy().reshape(-1, self.dim)

        # Fallback: random scalarization over objectives if EHVI is unavailable.
        weights = torch.rand(self.n_obj, dtype=torch.double)
        weights = weights / weights.sum().clamp_min(1e-12)
        scalar_Y = (train_Y * weights).sum(dim=1, keepdim=True)
        model = self.build_model(train_X, scalar_Y)
        acquisition = LogExpectedImprovement(model, best_f=scalar_Y.max().item())
        X_next, _ = optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return X_next.detach().cpu().numpy().reshape(-1, self.dim)

    def nondominated_mask_minimize(self, F):
        F = np.asarray(F, dtype=float)
        n = F.shape[0]
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def restore_raw_objectives(self, F_minimized):
        F = np.asarray(F_minimized, dtype=float).copy()
        for j, obj in enumerate(self.objectives):
            if str(obj.get("sense", "Minimize")).lower() == "maximize":
                F[:, j] *= -1.0
        return F

    def run(self):
        observed_keys = set()
        self.initial_design_multi(observed_keys)

        for iteration in range(self.n_iter):
            valid_records = self.get_valid_multi_records()
            if len(valid_records) < max(2, self.n_obj + 1):
                x_next = self.random_unobserved(observed_keys)
                if x_next is None:
                    break
                observed_keys.add(self.point_key(x_next))
                self.evaluate_and_store_multi(x_next)
                continue

            train_X, train_Y, _ = self.build_training_data_multi(valid_records)
            try:
                X_next_norm = self.propose_mobo_candidate(train_X, train_Y, observed_keys)
            except Exception as e:
                print(f"Error en MOBO acquisition: {e}", flush=True)
                X_next_norm = []

            evaluated_any = False
            for xn in X_next_norm:
                x_candidate = self.round_to_valid_x(self.denormalize_x(xn))
                key = self.point_key(x_candidate)
                if key in observed_keys or not self.satisfies_input_constraints(x_candidate):
                    x_candidate = self.random_unobserved(observed_keys)
                if x_candidate is None:
                    continue
                observed_keys.add(self.point_key(x_candidate))
                self.evaluate_and_store_multi(x_candidate)
                evaluated_any = True

            if not evaluated_any:
                x_candidate = self.random_unobserved(observed_keys)
                if x_candidate is None:
                    break
                observed_keys.add(self.point_key(x_candidate))
                self.evaluate_and_store_multi(x_candidate)

        valid_records = self.get_valid_multi_records()
        if len(valid_records) == 0:
            return {
                "status": "error",
                "message": "Multi-objective Bayesian Optimization terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "algorithm": self.config.get("algorithm"),
                "problem_type": self.config.get("problem_type"),
                "objective_names": self.objective_names,
                "pareto_F_minimized": [],
                "pareto_F_raw": [],
                "pareto_X": [],
                "history": self.history_records,
            }, self.history_records

        F = np.array([r["objective_values"] for r in valid_records], dtype=float)
        X = np.array([
            [r["x"][v["tag"]] for v in self.config["inputs"]]
            for r in valid_records
        ], dtype=float)
        nd_mask = self.nondominated_mask_minimize(F)
        pareto_F = F[nd_mask]
        pareto_X = X[nd_mask]
        pareto_F_raw = self.restore_raw_objectives(pareto_F)

        return {
            "status": "success",
            "message": "Continuous Multi-objective Bayesian Optimization finished using qEHVI when available.",
            "algorithm": self.config.get("algorithm"),
            "problem_type": self.config.get("problem_type"),
            "objective_names": self.objective_names,
            "pareto_F_minimized": pareto_F.tolist(),
            "pareto_F_raw": pareto_F_raw.tolist(),
            "pareto_X": pareto_X.tolist(),
            "history": self.history_records,
            "evaluations": self.history_records,
        }, self.history_records

