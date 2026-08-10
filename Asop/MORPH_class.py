# MORPH_class.py

import math
import numpy as np
import torch
from dataclasses import dataclass

from botorch.acquisition import LogExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

from bayesian_classes import VanillaBOBotorch, PENALTY_VALUE

try:
    from torch.quasirandom import SobolEngine
except Exception:
    SobolEngine = None




try:
    from botorch.utils.multi_objective.hypervolume import Hypervolume
except Exception:
    Hypervolume = None



# ============================================================
# MORPH-BO State
# ============================================================

@dataclass
class MorphState:
    dim: int
    batch_size: int = 1

    # Trust region continua en espacio normalizado [0, 1]
    length: float = 0.5
    length_min: float = 1e-6
    length_max: float = 1.0

    # Trust region discreta en unidades enteras reales
    discrete_radius: int = 3
    discrete_radius_min: int = 1
    discrete_radius_max: int = 10

    success_counter: int = 0
    failure_counter: int = 0

    success_tolerance: int = 3
    failure_tolerance: int = 0

    shrink_factor: float = 0.7
    expand_factor: float = 1.5

    best_value: float = -float("inf")
    restart_triggered: bool = False

    abs_tol: float = 1e-6
    rel_tol: float = 1e-3

    def __post_init__(self):
        if self.failure_tolerance is None or int(self.failure_tolerance) <= 0:
            self.failure_tolerance = max(
                3,
                math.ceil(self.dim / max(self.batch_size, 1))
            )


# ============================================================
# MORPH-BO with Feasibility Learning
# ============================================================

class MorphBOBotorch(VanillaBOBotorch):
    """
    MORPH-BO-F monoobjetivo para variables continuas y discretas.

    Características:
        1. BO monoobjetivo en el espacio original normalizado, sin VAE.
        2. Variables continuas + discretas/enteras.
        3. Trust region mixta:
            - longitud normalizada para variables continuas;
            - radio entero para variables discretas.
        4. Kernel mixto adaptativo con ARD:
            - Matérn 5/2 para continuas;
            - Matérn 5/2 ordinal para discretas normalizadas;
            - interacción continuo-discreto.
        5. Modelo de factibilidad:
            p_feas(x) = P(Aspen converge | x).
        6. Adquisición constrained LogEI:
            log(alpha_final) = LogEI(x) + rho * log(p_feas(x)),
            equivalente a EI(x) * p_feas(x)^rho.
    """

    def __init__(self, config, aspen):
        super().__init__(config, aspen)

        # MORPHBO monoobjetivo utiliza exclusivamente LogExpectedImprovement.
        self.acqf = "logei"

        self.batch_size = int(float(self.hyper.get("batch_size", 1)))

        self.n_init = int(float(
            self.hyper.get("n_init", max(2 * self.dim, 8))
        ))

        self.n_iter = int(float(
            self.hyper.get("n_iter", 30)
        ))

        self.n_candidates = int(float(
            self.hyper.get("n_candidates", max(2000, 200 * self.dim))
        ))

        self.global_candidate_fraction = float(
            self.hyper.get("global_candidate_fraction", 0.0)
        )

        # Continuous TR
        self.length = float(self.hyper.get("length", 0.5))
        self.length_min = float(self.hyper.get("length_min", 1e-6))
        self.length_max = float(self.hyper.get("length_max", 1.0))

        # Discrete TR
        self.discrete_radius = int(float(
            self.hyper.get("discrete_radius", 3)
        ))

        self.discrete_radius_min = int(float(
            self.hyper.get("discrete_radius_min", 1)
        ))

        self.discrete_radius_max = int(float(
            self.hyper.get("discrete_radius_max", 10)
        ))

        # TR update parameters
        self.success_tolerance = int(float(
            self.hyper.get("success_tolerance", 3)
        ))

        self.failure_tolerance = int(float(
            self.hyper.get("failure_tolerance", -1)
        ))

        self.shrink_factor = float(
            self.hyper.get("shrink_factor", 0.7)
        )

        self.expand_factor = float(
            self.hyper.get("expand_factor", 1.5)
        )

        self.abs_tol = float(
            self.hyper.get("abs_tol", 1e-6)
        )

        self.rel_tol = float(
            self.hyper.get("rel_tol", 1e-3)
        )


        # Feasibility model
        self.use_feasibility = str(
            self.hyper.get("use_feasibility", "true")
        ).strip().lower() in ["true", "1", "yes", "si", "sí"]

        self.feasibility_rho = float(
            self.hyper.get("feasibility_rho", 1.0)
        )

        self.p_feas_min = float(
            self.hyper.get("p_feas_min", 1e-3)
        )

        self.min_feasibility_points = int(float(
            self.hyper.get("min_feasibility_points", max(5, self.dim + 1))
        ))

        # Kernel
        self.kernel_mode = str(
            self.hyper.get("kernel_mode", "mixed")
        ).strip().lower()

        if np.any(self.ub <= self.lb):
            raise RuntimeError(
                "Error en bounds: todos los upper_bound deben ser mayores que lower_bound."
            )

        self.variable_types = [
            str(v.get("variable_type", "Continuous")).strip().lower()
            for v in self.config.get("inputs", [])
        ]

        allowed = {"continuous", "discrete", "integer"}

        unsupported = [
            v for v in self.variable_types
            if v not in allowed
        ]

        if unsupported:
            raise RuntimeError(
                "MORPH-BO-F monoobjetivo solo acepta variables Continuous y Discrete. "
                f"Tipos no soportados: {unsupported}"
            )

        self.continuous_idx = [
            i for i, t in enumerate(self.variable_types)
            if t == "continuous"
        ]

        self.discrete_idx = [
            i for i, t in enumerate(self.variable_types)
            if t in ["discrete", "integer"]
        ]

    # ---------------------------------------------------------
    # Sobol / random initial points
    # ---------------------------------------------------------

    def _sobol_points(self, n):
        if SobolEngine is not None:
            sobol = SobolEngine(
                self.dim,
                scramble=True,
                seed=self.seed
            )

            return sobol.draw(n).to(dtype=torch.double)

        return torch.tensor(
            self.rng.random((n, self.dim)),
            dtype=torch.double
        )

    def initial_design(self, observed_keys):
        X01 = self._sobol_points(
            max(self.n_init * 5, self.n_init)
        ).cpu().numpy()

        for xn in X01:
            x = self.round_to_valid_x(
                self.denormalize_x(xn)
            )

            key = self.point_key(x)

            if key in observed_keys:
                continue

            if not self.satisfies_input_constraints(x):
                continue

            observed_keys.add(key)
            self.evaluate_and_store(x)

            if len(self.history_records) >= self.n_init:
                break

        while len(self.history_records) < self.n_init:
            x = self.random_unobserved(observed_keys)

            if x is None:
                break

            observed_keys.add(self.point_key(x))
            self.evaluate_and_store(x)

    # ---------------------------------------------------------
    # Mixed kernel
    # ---------------------------------------------------------

    def build_adaptive_mixed_kernel(self):
        """
        Kernel mixto para variables continuas y discretas ordinales.

        Continuas:
            Matérn 5/2

        Discretas ordinales:
            Matérn 5/2 sobre valores enteros normalizados

        Interacción:
            Matérn_continuo * Matérn_discreto
        """

        if self.kernel_mode in ["matern", "matern_only", "standard"]:
            return ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.dim,
                    lengthscale_constraint=Interval(0.01, 10.0)
                )
            )

        kernels = []

        cont_kernel_for_sum = None
        disc_kernel_for_sum = None

        if len(self.continuous_idx) > 0:
            cont_kernel_for_sum = MaternKernel(
                nu=2.5,
                ard_num_dims=len(self.continuous_idx),
                active_dims=self.continuous_idx,
                lengthscale_constraint=Interval(0.01, 10.0)
            )

            kernels.append(cont_kernel_for_sum)

        if len(self.discrete_idx) > 0:
            disc_kernel_for_sum = MaternKernel(
                nu=2.5,
                ard_num_dims=len(self.discrete_idx),
                active_dims=self.discrete_idx,
                lengthscale_constraint=Interval(0.01, 10.0)
            )

            kernels.append(disc_kernel_for_sum)

        if len(kernels) == 0:
            raise RuntimeError("No hay variables de entrada para construir el kernel.")

        covar = kernels[0]

        for k in kernels[1:]:
            covar = covar + k

        # Interacción continuo-discreto ordinal
        if (
            len(self.continuous_idx) > 0
            and len(self.discrete_idx) > 0
            and self.kernel_mode in ["mixed", "hybrid", "morph"]
        ):
            cont_for_interaction = MaternKernel(
                nu=2.5,
                ard_num_dims=len(self.continuous_idx),
                active_dims=self.continuous_idx,
                lengthscale_constraint=Interval(0.01, 10.0)
            )

            disc_for_interaction = MaternKernel(
                nu=2.5,
                ard_num_dims=len(self.discrete_idx),
                active_dims=self.discrete_idx,
                lengthscale_constraint=Interval(0.01, 10.0)
            )

            interaction_kernel = (
                cont_for_interaction
                * disc_for_interaction
            )

            covar = covar + interaction_kernel

        return ScaleKernel(covar)


    # ---------------------------------------------------------
    # Objective model
    # ---------------------------------------------------------

    def get_valid_objective_records(self):
        return [
            r for r in self.history_records
            if r.get("feasible")
            and r.get("aspen_converged")
            and np.isfinite(r.get("objective", np.nan))
        ]

    def get_objective_training_data(self):
        valid_records = self.get_valid_objective_records()

        if len(valid_records) == 0:
            return None, None, []

        X_np = np.array([
            [r["x"][v["tag"]] for v in self.config["inputs"]]
            for r in valid_records
        ], dtype=float)

        # BoTorch maximiza.
        # Tu app guarda objetivos como minimización.
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

        return train_X, train_Y, valid_records

    def build_objective_model(self, train_X, train_Y):
        model = SingleTaskGP(
            train_X,
            train_Y,
            outcome_transform=Standardize(m=1),
            likelihood=GaussianLikelihood(
                noise_constraint=Interval(1e-6, 1e-1)
            ),
            covar_module=self.build_adaptive_mixed_kernel()
        )

        mll = ExactMarginalLogLikelihood(
            model.likelihood,
            model
        )

        fit_gpytorch_mll(mll)

        return model

    # ---------------------------------------------------------
    # Feasibility model
    # ---------------------------------------------------------

    def record_is_feasible_for_learning(self, record):
        return (
            record.get("feasible")
            and record.get("aspen_converged")
            and np.isfinite(record.get("objective", np.nan))
            and float(record.get("objective", PENALTY_VALUE)) < PENALTY_VALUE
        )

    def get_feasibility_training_data(self):
        records = self.history_records

        if len(records) < self.min_feasibility_points:
            return None, None, None

        X_np = np.array([
            [r["x"][v["tag"]] for v in self.config["inputs"]]
            for r in records
        ], dtype=float)

        y_feas = np.array([
            [1.0 if self.record_is_feasible_for_learning(r) else 0.0]
            for r in records
        ], dtype=float)

        mean_feas = float(np.mean(y_feas))

        # Si todos son factibles o todos no factibles,
        # no tiene sentido ajustar un GP de factibilidad todavía.
        if len(np.unique(y_feas.reshape(-1))) < 2:
            return None, None, mean_feas

        train_X = torch.tensor(
            np.array([self.normalize_x(x) for x in X_np]),
            dtype=torch.double
        )

        train_Y = torch.tensor(
            y_feas,
            dtype=torch.double
        )

        return train_X, train_Y, mean_feas

    def build_feasibility_model(self):
        if not self.use_feasibility:
            return None, 1.0

        train_X, train_Y, mean_feas = self.get_feasibility_training_data()

        if train_X is None or train_Y is None:
            if mean_feas is None:
                return None, 1.0

            return None, float(mean_feas)

        model = SingleTaskGP(
            train_X,
            train_Y,
            likelihood=GaussianLikelihood(
                noise_constraint=Interval(1e-5, 1e-1)
            ),
            covar_module=self.build_adaptive_mixed_kernel()
        )

        mll = ExactMarginalLogLikelihood(
            model.likelihood,
            model
        )

        fit_gpytorch_mll(mll)

        return model, float(mean_feas)

    def predict_feasibility(self, feasibility_model, default_feas, Xcand):
        if not self.use_feasibility:
            return torch.ones(
                Xcand.shape[0],
                dtype=torch.double,
                device=Xcand.device
            )

        if feasibility_model is None:
            return torch.full(
                (Xcand.shape[0],),
                fill_value=max(default_feas, self.p_feas_min),
                dtype=torch.double,
                device=Xcand.device
            )

        with torch.no_grad():
            posterior = feasibility_model.posterior(Xcand)
            mean = posterior.mean.reshape(-1)

        p_feas = mean.clamp(
            min=self.p_feas_min,
            max=1.0
        )

        return p_feas

    # ---------------------------------------------------------
    # Candidate pool
    # ---------------------------------------------------------

    def _append_candidate_if_valid(self, Xraw, x, candidate_keys, observed_keys):
        x = self.round_to_valid_x(x)
        key = self.point_key(x)

        if key in observed_keys:
            return False

        if key in candidate_keys:
            return False

        if not self.satisfies_input_constraints(x):
            return False

        Xraw.append(x)
        candidate_keys.add(key)

        return True

    def build_local_candidate_pool(self, x_center, state, observed_keys):
        x_center = self.round_to_valid_x(x_center)
        xn_center = self.normalize_x(x_center)

        Xraw = []
        candidate_keys = set()

        n_global = int(
            max(0, min(1, self.global_candidate_fraction))
            * self.n_candidates
        )

        n_local = self.n_candidates - n_global

        attempts = 0
        max_attempts = max(self.n_candidates * 80, 8000)

        # ----------------------------
        # Local candidates
        # ----------------------------
        while len(Xraw) < n_local and attempts < max_attempts:
            attempts += 1

            xn = np.array(xn_center, dtype=float).copy()
            x = np.array(x_center, dtype=float).copy()

            # Continuous variables
            for i in self.continuous_idx:
                lo = max(0.0, xn_center[i] - state.length / 2.0)
                hi = min(1.0, xn_center[i] + state.length / 2.0)

                xn[i] = self.rng.uniform(lo, hi)

            x_from_xn = self.denormalize_x(xn)

            for i in self.continuous_idx:
                x[i] = x_from_xn[i]

            # Discrete variables
            for i in self.discrete_idx:
                center_i = int(round(x_center[i]))
                lb_i = int(round(self.lb[i]))
                ub_i = int(round(self.ub[i]))

                r = int(max(state.discrete_radius, 1))

                lo_i = max(lb_i, center_i - r)
                hi_i = min(ub_i, center_i + r)

                if hi_i < lo_i:
                    lo_i, hi_i = lb_i, ub_i

                x[i] = int(
                    self.rng.integers(lo_i, hi_i + 1)
                )

            self._append_candidate_if_valid(
                Xraw=Xraw,
                x=x,
                candidate_keys=candidate_keys,
                observed_keys=observed_keys
            )

        # ----------------------------
        # Global candidates
        # ----------------------------
        # Los candidatos globales son opcionales. Con el valor metodológico
        # predeterminado global_candidate_fraction=0, todo el pool permanece
        # dentro de la trust region mixta.
        if n_global > 0:
            global_attempts = 0
            max_global_attempts = max(n_global * 80, 1000)

            while (
                len(Xraw) < self.n_candidates
                and global_attempts < max_global_attempts
            ):
                global_attempts += 1

                xn = self.rng.random(self.dim)
                x = self.round_to_valid_x(
                    self.denormalize_x(xn)
                )

                self._append_candidate_if_valid(
                    Xraw=Xraw,
                    x=x,
                    candidate_keys=candidate_keys,
                    observed_keys=observed_keys
                )

        if len(Xraw) == 0:
            x_fallback = self.random_unobserved(observed_keys)

            if x_fallback is None:
                return None, None

            Xraw = [x_fallback]

        Xraw = np.array(Xraw, dtype=float)

        Xn = np.array([
            self.normalize_x(x)
            for x in Xraw
        ], dtype=float)

        return torch.tensor(
            Xn,
            dtype=torch.double
        ), Xraw

    # ---------------------------------------------------------
    # Constrained LogEI acquisition
    # ---------------------------------------------------------

    def acquisition_scores(self, objective_model, train_Y, Xcand):
        """
        Calcula LogExpectedImprovement para cada candidato.

        BoTorch devuelve log(EI), por lo que la factibilidad se incorpora
        posteriormente como:

            LogEI(x) + rho * log(P_feas(x))

        Esto conserva la escala logarítmica y es equivalente a maximizar
        EI(x) * P_feas(x)^rho.
        """
        acquisition = LogExpectedImprovement(
            objective_model,
            best_f=train_Y.max().item()
        )

        with torch.no_grad():
            try:
                scores = acquisition(
                    Xcand.unsqueeze(1)
                ).reshape(-1)
            except Exception:
                scores = acquisition(
                    Xcand
                ).reshape(-1)

        return torch.nan_to_num(
            scores,
            nan=-1.0e20,
            posinf=1.0e20,
            neginf=-1.0e20
        )

    def combine_logei_with_feasibility(self, log_ei, p_feas):
        p_feas_safe = torch.nan_to_num(
            p_feas.reshape(-1),
            nan=self.p_feas_min,
            posinf=1.0,
            neginf=self.p_feas_min
        ).clamp(
            min=max(float(self.p_feas_min), 1e-12),
            max=1.0
        )

        return (
            log_ei.reshape(-1)
            + self.feasibility_rho * torch.log(p_feas_safe)
        )

    def propose_candidate(
        self,
        objective_model,
        feasibility_model,
        default_feas,
        train_Y,
        valid_records,
        state,
        observed_keys
    ):
        # La trust region monoobjetivo se centra en el mejor punto factible.
        best_record = min(
            valid_records,
            key=lambda r: r["objective"]
        )

        x_center = np.array([
            best_record["x"][v["tag"]]
            for v in self.config["inputs"]
        ], dtype=float)

        Xcand, Xraw = self.build_local_candidate_pool(
            x_center=x_center,
            state=state,
            observed_keys=observed_keys
        )

        if Xcand is None or Xraw is None:
            return self.random_unobserved(observed_keys)

        log_ei = self.acquisition_scores(
            objective_model=objective_model,
            train_Y=train_Y,
            Xcand=Xcand
        )

        p_feas = self.predict_feasibility(
            feasibility_model=feasibility_model,
            default_feas=default_feas,
            Xcand=Xcand
        )

        final_scores = self.combine_logei_with_feasibility(
            log_ei=log_ei,
            p_feas=p_feas
        )

        final_scores = torch.nan_to_num(
            final_scores,
            nan=-1.0e20,
            posinf=1.0e20,
            neginf=-1.0e20
        )

        order = torch.argsort(final_scores, descending=True)

        for idx in order:
            i = int(idx.item())
            x_try = self.round_to_valid_x(Xraw[i])
            key_try = self.point_key(x_try)

            if key_try in observed_keys:
                continue

            if not self.satisfies_input_constraints(x_try):
                continue

            return x_try

        return self.random_unobserved(observed_keys)

    # ---------------------------------------------------------
    # Trust-region update
    # ---------------------------------------------------------

    def update_state(self, state, y_next):
        y_next = float(y_next)

        if not np.isfinite(y_next):
            y_next = -PENALTY_VALUE

        if state.best_value == -float("inf"):
            state.best_value = y_next
            return state

        threshold = max(
            state.abs_tol,
            state.rel_tol * max(1.0, abs(state.best_value))
        )

        if y_next > state.best_value + threshold:
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter >= state.success_tolerance:
            state.length = min(
                state.length * state.expand_factor,
                state.length_max
            )

            state.discrete_radius = min(
                int(math.ceil(state.discrete_radius * state.expand_factor)),
                state.discrete_radius_max
            )

            state.success_counter = 0

        elif state.failure_counter >= state.failure_tolerance:
            state.length = max(
                state.length * state.shrink_factor,
                state.length_min
            )

            state.discrete_radius = max(
                int(math.floor(state.discrete_radius * state.shrink_factor)),
                state.discrete_radius_min
            )

            state.failure_counter = 0

        state.best_value = max(
            state.best_value,
            y_next
        )

        if (
            state.length <= state.length_min
            and state.discrete_radius <= state.discrete_radius_min
        ):
            state.restart_triggered = True

        return state

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------


    def run(self):
        observed_keys = set()

        state = MorphState(
            dim=self.dim,
            batch_size=self.batch_size,
            length=self.length,
            length_min=self.length_min,
            length_max=self.length_max,
            discrete_radius=self.discrete_radius,
            discrete_radius_min=self.discrete_radius_min,
            discrete_radius_max=self.discrete_radius_max,
            success_tolerance=self.success_tolerance,
            failure_tolerance=self.failure_tolerance,
            shrink_factor=self.shrink_factor,
            expand_factor=self.expand_factor,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol
        )

        # Initial evaluations
        self.initial_design(observed_keys)

        for iteration in range(self.n_iter):
            train_X, train_Y, valid_records = self.get_objective_training_data()

            # Si todavía no hay suficientes puntos válidos,
            # seguimos explorando aleatoriamente.
            if train_X is None or len(valid_records) < 2:
                x_next = self.random_unobserved(observed_keys)

                if x_next is None:
                    break

                observed_keys.add(self.point_key(x_next))
                record = self.evaluate_and_store(x_next)

                if self.record_is_feasible_for_learning(record):
                    y_next = -float(record["objective"])
                else:
                    y_next = -PENALTY_VALUE

                state = self.update_state(state, y_next)
                continue

            try:
                objective_model = self.build_objective_model(
                    train_X,
                    train_Y
                )

                feasibility_model, default_feas = self.build_feasibility_model()

                x_candidate = self.propose_candidate(
                    objective_model=objective_model,
                    feasibility_model=feasibility_model,
                    default_feas=default_feas,
                    train_Y=train_Y,
                    valid_records=valid_records,
                    state=state,
                    observed_keys=observed_keys
                )

            except Exception as e:
                print(
                    f"Error en MORPH-BO-F acquisition/model: {e}",
                    flush=True
                )

                x_candidate = self.random_unobserved(observed_keys)

            if x_candidate is None:
                break

            candidate_key = self.point_key(x_candidate)

            if (
                candidate_key in observed_keys
                or not self.satisfies_input_constraints(x_candidate)
            ):
                x_candidate = self.random_unobserved(observed_keys)

            if x_candidate is None:
                break

            observed_keys.add(self.point_key(x_candidate))
            record = self.evaluate_and_store(x_candidate)

            if self.record_is_feasible_for_learning(record):
                y_next = -float(record["objective"])
            else:
                y_next = -PENALTY_VALUE

            state = self.update_state(state, y_next)

            if state.restart_triggered:
                print(
                    "MORPH-BO-F trust region reached minimum size.",
                    flush=True
                )
                break

        valid_records = self.get_valid_objective_records()

        if len(valid_records) == 0:
            return {
                "status": "error",
                "message": "MORPH-BO-F terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "best_x": None,
                "best_objective": None,
                "outputs": None,
                "evaluations": self.history_records,
                "trust_region_length": state.length,
                "discrete_radius": state.discrete_radius,
                "feasibility_model_used": bool(self.use_feasibility)
            }, self.history_records

        best_record = min(
            valid_records,
            key=lambda r: r["objective"]
        )

        return {
            "status": "success",
            "message": "MORPH-BO-F single-objective optimization finished using constrained LogExpectedImprovement.",
            "best_x": best_record["x"],
            "best_objective": float(best_record["objective"]),
            "outputs": best_record["outputs"],
            "evaluations": self.history_records,
            "trust_region_length": state.length,
            "discrete_radius": state.discrete_radius,
            "restart_triggered": bool(state.restart_triggered),
            "feasibility_model_used": bool(self.use_feasibility),
            "kernel_mode": self.kernel_mode
        }, self.history_records


# ============================================================
# MORPH-BO-F Multi-objective
# ============================================================

class MorphMOBOBotorch(MorphBOBotorch):
    """
    MORPH-BO-F multiobjetivo para variables continuas y discretas.

    Convenciones:
        - La aplicación guarda F en convención de minimización.
        - Los GP trabajan internamente con Y = -F porque BoTorch maximiza.
        - No se utiliza VAE ni espacio latente.

    Metodología:
        1. Genera candidatos dentro de una trust region continua-discreta.
        2. Ajusta un GP independiente por objetivo con kernel mixto adaptativo.
        3. Predice la media y desviación estándar de cada objetivo.
        4. Estima un frente de Pareto usando las medias predichas.
        5. En ese frente, calcula incertidumbre conjunta normalizada.
        6. Pondera la incertidumbre por P_feas(x)^rho.
        7. Evalúa en Aspen el candidato con mayor puntuación.

    Adquisición:
        alpha_MO(x) = I[x pertenece al Pareto predicho]
                      * U_joint(x) * P_feas(x)^rho.
    """

    def __init__(self, config, aspen):
        super().__init__(config, aspen)

        self.objectives = config.get("objectives", []) or []

        if len(self.objectives) < 2:
            raise RuntimeError(
                "MORPHBO multiobjetivo requiere al menos dos funciones objetivo."
            )

        self.objective_names = [
            str(
                obj.get("name")
                or obj.get("tag")
                or obj.get("alias")
                or f"obj{i + 1}"
            )
            for i, obj in enumerate(self.objectives)
        ]

        self.n_obj = len(self.objectives)

        self.mobo_acqf = "predicted_pareto_uncertainty_feasibility"

        self.ref_point_raw = self.hyper.get(
            "ref_point",
            self.hyper.get("reference_point", "auto")
        )

        # Multi-objective trust-region center. By default, the center is the
        # observed Pareto design with the minimum normalized Euclidean distance
        # to the observed ideal point. Equal objective weights are used unless
        # center_weights is explicitly provided.
        self.center_strategy = str(
            self.hyper.get("center_strategy", "ideal_distance")
        ).strip().lower()

        self.center_epsilon = float(
            self.hyper.get("center_epsilon", 1e-12)
        )

        self.center_weights = self.parse_center_weights(
            self.hyper.get("center_weights", "equal")
        )

        self._last_center_info = None
        self._fixed_ref_point_y = None

        # Para MOBO usualmente conviene un n_init un poco mayor.
        self.n_init = int(float(
            self.hyper.get(
                "n_init",
                max(2 * self.dim + 2, 3 * self.n_obj + 4, 10)
            )
        ))

        self.n_iter = int(float(
            self.hyper.get("n_iter", 30)
        ))

        self.n_candidates = int(float(
            self.hyper.get("n_candidates", max(3000, 300 * self.dim))
        ))

    # ---------------------------------------------------------
    # Multi-objective Aspen evaluation
    # ---------------------------------------------------------

    def evaluate_point_multi(self, aspen, config, x):
        x = self.preprocess_x(config, x)

        ok_constraints, violations = self.check_input_constraints(config, x)

        if not ok_constraints:
            return self.penalty_result_multi(
                config=config,
                x=x,
                reason="Punto rechazado antes de correr Aspen porque no cumple restricciones de entrada.",
                violations=violations
            )

        try:
            self.update_aspen(aspen, config)

            for input_var, value in zip(config["inputs"], x):
                self.set_aspen_variable(aspen, input_var["path"], value)

            self.run_aspen(aspen)

            converged, run_status, status_message = self.check_aspen_convergence(aspen)

            if not converged:
                self.reinit_aspen(aspen)
                return self.penalty_result_multi(
                    config=config,
                    x=x,
                    reason=status_message,
                    run_status=run_status
                )

            outputs = {}
            for output_var in config.get("outputs", []):
                outputs[output_var["tag"]] = self.get_aspen_variable(
                    aspen,
                    output_var["path"]
                )

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
                    raise RuntimeError(
                        f"La función objetivo no es finita: {value_raw}"
                    )

                sense = str(obj.get("sense", "Minimize")).strip().lower()

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
            return self.penalty_result_multi(
                config=config,
                x=x,
                reason=f"Error al evaluar el punto en Aspen: {e}"
            )

    def penalty_result_multi(self, config, x, reason, violations=None, run_status=None):
        x = self.preprocess_x(config, x)

        outputs = {
            output_var["tag"]: None
            for output_var in config.get("outputs", [])
        }

        return {
            "objective_values": [PENALTY_VALUE for _ in self.objectives],
            "objective_values_raw": [None for _ in self.objectives],
            "outputs": outputs,
            "x": x,
            "feasible": False,
            "aspen_converged": False,
            "run_status": run_status,
            "message": reason,
            "violations": violations or []
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
            "algorithm": "MORPHBO",
            "mobo_acqf": self.mobo_acqf,
            "trust_region_length": None,
            "discrete_radius": None,
            "candidate_source": None,
            "hypervolume": None,
            "tr_center_x": None,
            "tr_center_distance": None,
            "tr_center_weights": None,
            "tr_center_ideal_point": None,
            "tr_center_nadir_point": None,
        }

        self.history_records.append(record)
        return record

    def initial_design_multi(self, observed_keys):
        X01 = self._sobol_points(
            max(self.n_init * 5, self.n_init)
        ).cpu().numpy()

        for xn in X01:
            x = self.round_to_valid_x(
                self.denormalize_x(xn)
            )

            key = self.point_key(x)

            if key in observed_keys:
                continue

            if not self.satisfies_input_constraints(x):
                continue

            observed_keys.add(key)
            record = self.evaluate_and_store_multi(x)
            record["candidate_source"] = "initial_sobol"

            if len(self.history_records) >= self.n_init:
                break

        while len(self.history_records) < self.n_init:
            x = self.random_unobserved(observed_keys)

            if x is None:
                break

            observed_keys.add(self.point_key(x))
            record = self.evaluate_and_store_multi(x)
            record["candidate_source"] = "initial_random"

    # ---------------------------------------------------------
    # Records, Pareto and training data
    # ---------------------------------------------------------

    def record_is_feasible_for_learning(self, record):
        objective_values = record.get("objective_values", []) or []

        return (
            record.get("feasible")
            and record.get("aspen_converged")
            and len(objective_values) == self.n_obj
            and all(np.isfinite(v) for v in objective_values)
            and all(float(v) < PENALTY_VALUE for v in objective_values)
        )

    def get_valid_multi_records(self):
        return [
            r for r in self.history_records
            if self.record_is_feasible_for_learning(r)
        ]

    def build_training_data_multi(self, valid_records):
        X_np = np.array([
            [r["x"][v["tag"]] for v in self.config["inputs"]]
            for r in valid_records
        ], dtype=float)

        F_np = np.array([
            r["objective_values"]
            for r in valid_records
        ], dtype=float)

        train_X = torch.tensor(
            np.array([self.normalize_x(x) for x in X_np]),
            dtype=torch.double
        )

        # BoTorch maximiza. La app guarda F como minimización.
        train_Y = torch.tensor(
            -F_np,
            dtype=torch.double
        )

        return train_X, train_Y, F_np

    def pareto_mask_min(self, F):
        F = np.asarray(F, dtype=float)
        n = F.shape[0]
        mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not mask[i]:
                continue

            dominated_by_any = np.any(
                np.all(F <= F[i], axis=1)
                & np.any(F < F[i], axis=1)
            )

            if dominated_by_any:
                mask[i] = False

        return mask

    def pareto_mask_max_torch(self, Y):
        Y_np = Y.detach().cpu().numpy()
        F_np = -Y_np
        mask_np = self.pareto_mask_min(F_np)
        return torch.tensor(mask_np, dtype=torch.bool, device=Y.device)

    def get_pareto_records(self, valid_records):
        if len(valid_records) == 0:
            return []

        F = np.array([
            r["objective_values"]
            for r in valid_records
        ], dtype=float)

        mask = self.pareto_mask_min(F)

        return [
            r for r, keep in zip(valid_records, mask)
            if keep
        ]

    # ---------------------------------------------------------
    # GP model list with mixed kernel
    # ---------------------------------------------------------

    def build_model_list_multi(self, train_X, train_Y):
        models = []

        for j in range(train_Y.shape[1]):
            model_j = SingleTaskGP(
                train_X,
                train_Y[:, j:j + 1],
                outcome_transform=Standardize(m=1),
                likelihood=GaussianLikelihood(
                    noise_constraint=Interval(1e-6, 1e-1)
                ),
                covar_module=self.build_adaptive_mixed_kernel()
            )
            models.append(model_j)

        model = ModelListGP(*models)

        mll = SumMarginalLogLikelihood(
            model.likelihood,
            model
        )

        fit_gpytorch_mll(mll)

        return model

    # ---------------------------------------------------------
    # Reference point and hypervolume
    # ---------------------------------------------------------

    def parse_ref_point_y(self, train_Y):
        raw = str(self.ref_point_raw).strip().lower()

        if raw in ["", "auto", "none"]:
            y_min = train_Y.min(dim=0).values
            y_max = train_Y.max(dim=0).values
            margin = 0.10 * torch.clamp(y_max - y_min, min=1e-6)
            return (y_min - margin).to(dtype=torch.double)

        try:
            values = [
                float(v.strip())
                for v in str(self.ref_point_raw).replace(";", ",").split(",")
                if v.strip()
            ]

            if len(values) != self.n_obj:
                raise ValueError

            # Se interpreta en espacio de maximización Y = -F.
            return torch.tensor(values, dtype=torch.double)

        except Exception:
            raise RuntimeError(
                f"ref_point debe ser 'auto' o una lista de {self.n_obj} valores "
                "en espacio de maximización Y=-F."
            )

    def get_reference_point_y(self, train_Y):
        if self._fixed_ref_point_y is None:
            self._fixed_ref_point_y = self.parse_ref_point_y(train_Y)

        return self._fixed_ref_point_y.to(dtype=torch.double)

    def hypervolume_y(self, Y, ref_point_y):
        if Y is None or Y.numel() == 0:
            return 0.0

        if Hypervolume is None:
            # Fallback simple: métrica escalar positiva respecto al punto de referencia.
            improvement = torch.clamp(Y - ref_point_y, min=0.0)
            return float(torch.sum(torch.prod(improvement, dim=1)).item())

        mask = self.pareto_mask_max_torch(Y)
        pareto_Y = Y[mask]

        if pareto_Y.numel() == 0:
            return 0.0

        hv = Hypervolume(ref_point=ref_point_y)
        return float(hv.compute(pareto_Y).item())

    def current_hypervolume_from_records(self, valid_records):
        if len(valid_records) < 1:
            return 0.0

        _, train_Y, _ = self.build_training_data_multi(valid_records)
        ref_point_y = self.get_reference_point_y(train_Y)
        return self.hypervolume_y(train_Y, ref_point_y)

    # ---------------------------------------------------------
    # Candidate center and acquisitions
    # ---------------------------------------------------------

    def parse_center_weights(self, raw_weights):
        """
        Parse the objective weights used to select the multi-objective
        trust-region center.

        The default value ``equal`` assigns w_j = 1 / m to every objective.
        A comma-separated string or a numerical sequence can also be supplied.
        The weights are normalized so that their sum is one.
        """
        if raw_weights is None:
            return np.full(self.n_obj, 1.0 / self.n_obj, dtype=float)

        if isinstance(raw_weights, str):
            raw_text = raw_weights.strip().lower()

            if raw_text in ["", "equal", "uniform", "default", "none"]:
                return np.full(self.n_obj, 1.0 / self.n_obj, dtype=float)

            values = [
                float(value.strip())
                for value in raw_weights.replace(";", ",").split(",")
                if value.strip()
            ]
            weights = np.asarray(values, dtype=float)
        else:
            weights = np.asarray(raw_weights, dtype=float).reshape(-1)

        if weights.size != self.n_obj:
            raise RuntimeError(
                f"center_weights debe contener exactamente {self.n_obj} valores."
            )

        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise RuntimeError(
                "center_weights debe contener valores finitos y no negativos."
            )

        weight_sum = float(np.sum(weights))

        if weight_sum <= 0.0:
            raise RuntimeError(
                "La suma de center_weights debe ser mayor que cero."
            )

        return weights / weight_sum

    def normalized_ideal_distance_center(self, pareto_records):
        """
        Select the observed Pareto design closest to the observed ideal point.

        All objectives are assumed to be stored in the minimization convention.
        The Pareto objective values are normalized using the observed ideal and
        nadir approximations. The selected center minimizes

            d(x) = sqrt(sum_j w_j * f_tilde_j(x)^2).
        """
        if len(pareto_records) == 0:
            raise RuntimeError(
                "No hay registros Pareto disponibles para seleccionar el centro."
            )

        F = np.asarray(
            [record["objective_values"] for record in pareto_records],
            dtype=float
        )

        ideal_point = np.min(F, axis=0)
        nadir_point = np.max(F, axis=0)
        objective_ranges = nadir_point - ideal_point

        active_objectives = objective_ranges > max(self.center_epsilon, 0.0)
        F_normalized = np.zeros_like(F, dtype=float)

        if np.any(active_objectives):
            F_normalized[:, active_objectives] = (
                F[:, active_objectives]
                - ideal_point[active_objectives]
            ) / objective_ranges[active_objectives]

        distances = np.sqrt(
            np.sum(
                self.center_weights.reshape(1, -1)
                * np.square(F_normalized),
                axis=1
            )
        )

        center_index = int(np.argmin(distances))
        center_record = pareto_records[center_index]

        self._last_center_info = {
            "strategy": "ideal_distance",
            "distance": float(distances[center_index]),
            "weights": self.center_weights.tolist(),
            "ideal_point": ideal_point.tolist(),
            "nadir_point": nadir_point.tolist(),
            "normalized_objectives": F_normalized[center_index].tolist(),
            "center_x": dict(center_record["x"]),
        }

        return center_record

    def choose_tr_center_multi(self, valid_records):
        """
        Select the center of the multi-objective trust region.

        The default and recommended strategy is ``ideal_distance``: among the
        observed Pareto designs, select the design with the minimum normalized
        Euclidean distance to the observed ideal point. Equal weights are used
        by default, providing a centered and reproducible search around a
        balanced Pareto compromise.
        """
        pareto_records = self.get_pareto_records(valid_records)

        if len(pareto_records) == 0:
            pareto_records = valid_records

        if len(pareto_records) == 0:
            raise RuntimeError(
                "No hay puntos factibles para definir el centro multiobjetivo."
            )

        if self.center_strategy in [
            "ideal",
            "ideal_distance",
            "utopian",
            "utopian_distance",
            "balanced",
            "centered",
            "weighted_ideal",
            "equal_weight_ideal",
        ]:
            return self.normalized_ideal_distance_center(pareto_records)

        # Backward-compatible optional strategies. They are not the default
        # MORPHBO methodology.
        if self.center_strategy in ["random", "random_pareto", "pareto_random"]:
            idx = int(self.rng.integers(0, len(pareto_records)))
            center_record = pareto_records[idx]
            self._last_center_info = {
                "strategy": "random_pareto",
                "distance": None,
                "weights": self.center_weights.tolist(),
                "ideal_point": None,
                "nadir_point": None,
                "normalized_objectives": None,
                "center_x": dict(center_record["x"]),
            }
            return center_record

        if self.center_strategy in ["last", "last_pareto"]:
            center_record = pareto_records[-1]
            self._last_center_info = {
                "strategy": "last_pareto",
                "distance": None,
                "weights": self.center_weights.tolist(),
                "ideal_point": None,
                "nadir_point": None,
                "normalized_objectives": None,
                "center_x": dict(center_record["x"]),
            }
            return center_record

        raise RuntimeError(
            "center_strategy no reconocido. Use 'ideal_distance', "
            "'random_pareto' o 'last_pareto'."
        )

    def posterior_mean_std_multi(self, model, Xcand):
        """
        Devuelve medias y desviaciones estándar en convención de minimización F.

        Se itera explícitamente sobre los modelos para mantener compatibilidad
        entre versiones de BoTorch y ModelListGP.
        """
        means_y = []
        stds_y = []

        with torch.no_grad():
            for model_j in model.models:
                posterior_j = model_j.posterior(Xcand)
                means_y.append(posterior_j.mean.reshape(-1))
                stds_y.append(
                    posterior_j.variance
                    .clamp_min(1e-12)
                    .sqrt()
                    .reshape(-1)
                )

        mean_y = torch.stack(means_y, dim=1)
        std_y = torch.stack(stds_y, dim=1)

        # Y = -F; la incertidumbre no cambia con el signo.
        mean_f = -mean_y
        std_f = std_y

        return mean_f, std_f

    def predicted_pareto_uncertainty_scores(
        self,
        model,
        train_Y,
        Xcand,
        p_feas
    ):
        """
        Estima el frente de Pareto con las medias predichas y puntúa únicamente
        los candidatos no dominados mediante incertidumbre conjunta factible.
        """
        mean_f, std_f = self.posterior_mean_std_multi(
            model=model,
            Xcand=Xcand
        )

        pareto_mask_np = self.pareto_mask_min(
            mean_f.detach().cpu().numpy()
        )

        pareto_mask = torch.tensor(
            pareto_mask_np,
            dtype=torch.bool,
            device=Xcand.device
        )

        # Normalización por el rango observado para que un objetivo con unidades
        # grandes no domine la incertidumbre conjunta.
        objective_scale = (
            train_Y.max(dim=0).values
            - train_Y.min(dim=0).values
        ).abs().clamp_min(1e-8)

        std_normalized = std_f / objective_scale.reshape(1, -1)

        joint_uncertainty = torch.sqrt(
            torch.sum(std_normalized.pow(2), dim=1)
        )

        p_feas_safe = torch.nan_to_num(
            p_feas.reshape(-1),
            nan=self.p_feas_min,
            posinf=1.0,
            neginf=self.p_feas_min
        ).clamp(
            min=max(float(self.p_feas_min), 1e-12),
            max=1.0
        )

        final_scores = (
            joint_uncertainty
            * p_feas_safe.pow(self.feasibility_rho)
        )

        # Solo los candidatos del frente de Pareto predicho son elegibles.
        final_scores = torch.where(
            pareto_mask,
            final_scores,
            torch.full_like(final_scores, -torch.inf)
        )

        return torch.nan_to_num(
            final_scores,
            nan=-1.0e20,
            posinf=1.0e20,
            neginf=-1.0e20
        )

    def propose_mobo_candidate(
        self,
        model,
        feasibility_model,
        default_feas,
        train_Y,
        valid_records,
        state,
        observed_keys
    ):
        # The multi-objective trust region is centered, by default, at the
        # observed Pareto design closest to the normalized observed ideal point.
        center_record = self.choose_tr_center_multi(valid_records)

        x_center = np.array([
            center_record["x"][v["tag"]]
            for v in self.config["inputs"]
        ], dtype=float)

        Xcand, Xraw = self.build_local_candidate_pool(
            x_center=x_center,
            state=state,
            observed_keys=observed_keys
        )

        if Xcand is None or Xraw is None:
            return self.random_unobserved(observed_keys)

        p_feas = self.predict_feasibility(
            feasibility_model=feasibility_model,
            default_feas=default_feas,
            Xcand=Xcand
        )

        final_scores = self.predicted_pareto_uncertainty_scores(
            model=model,
            train_Y=train_Y,
            Xcand=Xcand,
            p_feas=p_feas
        )

        order = torch.argsort(final_scores, descending=True)

        for idx in order:
            i = int(idx.item())
            x_try = self.round_to_valid_x(Xraw[i])
            key_try = self.point_key(x_try)

            if key_try in observed_keys:
                continue

            if not self.satisfies_input_constraints(x_try):
                continue

            return x_try

        return self.random_unobserved(observed_keys)

    # ---------------------------------------------------------
    # Main MOBO loop
    # ---------------------------------------------------------

    def run(self):
        observed_keys = set()

        state = MorphState(
            dim=self.dim,
            batch_size=self.batch_size,
            length=self.length,
            length_min=self.length_min,
            length_max=self.length_max,
            discrete_radius=self.discrete_radius,
            discrete_radius_min=self.discrete_radius_min,
            discrete_radius_max=self.discrete_radius_max,
            success_tolerance=self.success_tolerance,
            failure_tolerance=self.failure_tolerance,
            shrink_factor=self.shrink_factor,
            expand_factor=self.expand_factor,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol
        )

        self.initial_design_multi(observed_keys)

        valid_records = self.get_valid_multi_records()

        if len(valid_records) >= max(2, self.n_obj + 1):
            state.best_value = self.current_hypervolume_from_records(valid_records)

        for iteration in range(self.n_iter):
            valid_records = self.get_valid_multi_records()

            if len(valid_records) < max(2, self.n_obj + 1):
                x_next = self.random_unobserved(observed_keys)

                if x_next is None:
                    break

                observed_keys.add(self.point_key(x_next))
                record = self.evaluate_and_store_multi(x_next)
                record["candidate_source"] = "random_until_enough_valid"
                record["trust_region_length"] = state.length
                record["discrete_radius"] = state.discrete_radius

                hv_now = self.current_hypervolume_from_records(
                    self.get_valid_multi_records()
                )
                record["hypervolume"] = hv_now
                state = self.update_state(state, hv_now)
                continue

            try:
                train_X, train_Y, _ = self.build_training_data_multi(valid_records)
                model = self.build_model_list_multi(
                    train_X=train_X,
                    train_Y=train_Y
                )

                feasibility_model, default_feas = self.build_feasibility_model()

                x_candidate = self.propose_mobo_candidate(
                    model=model,
                    feasibility_model=feasibility_model,
                    default_feas=default_feas,
                    train_Y=train_Y,
                    valid_records=valid_records,
                    state=state,
                    observed_keys=observed_keys
                )

            except Exception as e:
                print(
                    f"Error en MORPHBO multiobjetivo model/acquisition: {e}",
                    flush=True
                )
                x_candidate = self.random_unobserved(observed_keys)

            if x_candidate is None:
                break

            candidate_key = self.point_key(x_candidate)

            if (
                candidate_key in observed_keys
                or not self.satisfies_input_constraints(x_candidate)
            ):
                x_candidate = self.random_unobserved(observed_keys)

            if x_candidate is None:
                break

            observed_keys.add(self.point_key(x_candidate))
            record = self.evaluate_and_store_multi(x_candidate)
            record["candidate_source"] = "predicted_pareto_joint_uncertainty_pfeas_tr"
            record["trust_region_length"] = state.length
            record["discrete_radius"] = state.discrete_radius

            center_info = self._last_center_info or {}
            record["tr_center_x"] = center_info.get("center_x")
            record["tr_center_distance"] = center_info.get("distance")
            record["tr_center_weights"] = center_info.get("weights")
            record["tr_center_ideal_point"] = center_info.get("ideal_point")
            record["tr_center_nadir_point"] = center_info.get("nadir_point")

            hv_now = self.current_hypervolume_from_records(
                self.get_valid_multi_records()
            )
            record["hypervolume"] = hv_now

            state = self.update_state(state, hv_now)

            if state.restart_triggered:
                print(
                    "MORPHBO multiobjetivo trust region reached minimum size.",
                    flush=True
                )
                break

        valid_records = self.get_valid_multi_records()

        if len(valid_records) == 0:
            return {
                "status": "error",
                "message": "MORPHBO multiobjetivo terminó, pero ningún punto factible produjo una simulación Aspen convergida.",
                "pareto_X": [],
                "pareto_front": [],
                "pareto_front_raw": [],
                "evaluations": self.history_records,
                "trust_region_length": state.length,
                "discrete_radius": state.discrete_radius,
                "feasibility_model_used": bool(self.use_feasibility),
                "kernel_mode": self.kernel_mode,
                "mobo_acqf": self.mobo_acqf,
                "center_strategy": self.center_strategy,
                "center_weights": self.center_weights.tolist(),
            }, self.history_records

        train_X, train_Y, F_np = self.build_training_data_multi(valid_records)
        ref_point_y = self.get_reference_point_y(train_Y)
        hv_final = self.hypervolume_y(train_Y, ref_point_y)

        pareto_records = self.get_pareto_records(valid_records)

        pareto_X = [r["x"] for r in pareto_records]
        pareto_F = [r["objective_values"] for r in pareto_records]
        pareto_F_raw = [r.get("objective_values_raw") for r in pareto_records]
        pareto_outputs = [r.get("outputs") for r in pareto_records]

        return {
            "status": "success",
            "message": "MORPHBO multi-objective optimization finished using predicted Pareto means, joint uncertainty and feasibility.",
            "objective_names": self.objective_names,
            "pareto_X": pareto_X,
            "pareto_front": pareto_F,
            "pareto_front_raw": pareto_F_raw,
            "pareto_outputs": pareto_outputs,
            "hypervolume": hv_final,
            "reference_point_Y": ref_point_y.tolist(),
            "reference_point_F_minimization": (-ref_point_y).tolist(),
            "evaluations": self.history_records,
            "trust_region_length": state.length,
            "discrete_radius": state.discrete_radius,
            "restart_triggered": bool(state.restart_triggered),
            "feasibility_model_used": bool(self.use_feasibility),
            "kernel_mode": self.kernel_mode,
            "mobo_acqf": self.mobo_acqf,
            "center_strategy": self.center_strategy,
            "center_weights": self.center_weights.tolist(),
            "last_tr_center": self._last_center_info,
        }, self.history_records

