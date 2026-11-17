"""Incremental-iterative solution: Newton-Raphson + Newmark (dissertation Sections 4.5-4.7).

For every step the program searches the displacement increment that brings the
structure to equilibrium (Eq. 27):

    K_ef  du = F_ef - F_int

* static steps:   K_ef = K_t,                    F_ef = R (external loads)
* dynamic steps:  K_ef = a0 M + a1 D + K_t       (Eq. 28 / 48)
                  F_ef = R + M G + D E           (Eq. 29 / 49-50)

and iterates (Newton-Raphson) until the convergence criteria of Section 4.6
are satisfied. The numbered comments below follow the steps of Section 4.7.
"""

from __future__ import annotations

import time as clock
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .beam import Beams
from .contact import Contact
from .loads import StepInfo, load_factor
from .model import NDOF, Corrections, Model, Settings
from .results import Result


def newmark_constants(beta: float, gamma: float, dt: float) -> dict:
    """Integration constants a0..a7 (Eq. 31 and below Eq. 33)."""
    return {
        "a0": 1 / (beta * dt**2),
        "a1": gamma / (beta * dt),
        "a2": 1 / (beta * dt),
        "a3": 1 / (2 * beta) - 1,
        "a4": gamma / beta - 1,
        "a5": dt * (gamma / (2 * beta) - 1),
        "a6": (1 - gamma) * dt,
        "a7": gamma * dt,
    }


@dataclass
class Iteration:
    """Information passed to the optional ``on_iteration`` callback."""

    step: int
    iteration: int
    residual: np.ndarray  # R_F = F_ef - F_int (Eq. 37)
    internal_force: np.ndarray  # F_int
    du: np.ndarray  # displacement increment of this iteration
    disp_error: float  # left-hand side of Eq. 34
    force_error: float  # left-hand side of Eq. 35


def solve(
    model: Model,
    settings: Settings | None = None,
    corrections: Corrections | None = None,
    on_iteration: Callable[[Iteration], None] | None = None,
    on_step: Callable[[int, int, int], None] | None = None,
) -> Result:
    """Run the analysis and return the results of every saved step."""
    settings = settings or Settings()
    corrections = corrections or Corrections()
    start = clock.perf_counter()

    beams = Beams(model, corrections)
    contact = Contact(model, corrections.symmetric_contact_tangent)
    n = model.n_dofs

    # Row/column of every entry of the element matrices in the global matrix
    rows = np.repeat(beams.dofs, 12, axis=1).ravel()
    cols = np.tile(beams.dofs, (1, 12)).ravel()

    def assemble(element_matrices: np.ndarray) -> sp.csr_matrix:
        # Entries with the same (row, col) are summed when converting to CSR.
        return sp.coo_matrix((element_matrices.ravel(), (rows, cols)), shape=(n, n)).tocsr()

    # Prescribed DOFs (supports) and free DOFs
    fixed = np.unique([NDOF * s.node + s.dof for s in model.supports]).astype(int)
    free = np.setdiff1d(np.arange(n), fixed)
    self_weight = model.self_weight()

    n_static, n_dynamic = settings.n_static_steps, settings.n_dynamic_steps
    n_steps = n_static + n_dynamic
    a = newmark_constants(settings.beta, settings.gamma, settings.dt)
    alpha, beta = model.rayleigh_alpha, model.rayleigh_beta

    result = Result.empty(model, settings)

    # 2. initial displacements, velocities and accelerations (Eq. 38)
    U = np.zeros(n)
    V = np.zeros(n)
    Acc = np.zeros(n)

    # 3. loop over the steps
    for step in range(1, n_steps + 1):
        dynamic = step > n_static
        t = (step - n_static) * settings.dt if dynamic else 0.0
        info = StepInfo(
            step=step,
            n_static=n_static,
            dt=settings.dt,
            time=t,
            static_ratio=min(step / n_static, 1.0),
            ramp_ratio=min(t / settings.ramp_time, 1.0) if dynamic else 0.0,
            has_dynamic_phase=n_dynamic > 0,
        )

        # External loads R and target values of the prescribed DOFs at this step
        R_ext = np.zeros(n)
        for load in model.loads:
            R_ext[NDOF * load.node + load.dof] = load.value * load_factor(
                load.history, load.t_switch, info)
        R_ext += self_weight * load_factor(model.self_weight_history, 0.0, info)
        U_target = np.zeros(n)
        for s in model.supports:
            U_target[NDOF * s.node + s.dof] = s.value * load_factor(s.history, s.t_switch, info)

        du = np.zeros(n)  # increment of the last iteration
        dU = np.zeros(n)  # accumulated increment of the step (Eq. 36)
        R1_norm = 0.0
        converged = False

        # 4. Newton-Raphson iterations (at most max_iter)
        for it in range(1, settings.max_iter + 1):
            # 4.1-4.4 update element geometry and internal forces with du
            beams.update(du)
            F_int = beams.internal_force_vector()
            K_t = assemble(beams.tangent_stiffness())
            if contact.active:
                f_c, K_c = contact.forces_and_stiffness(U)
                F_int += f_c
                K_t = K_t + K_c

            if dynamic:
                M = assemble(beams.mass_matrices())
                D = alpha * M + beta * K_t  # Rayleigh damping, Eq. 20
                G = -a["a0"] * dU + a["a2"] * V + a["a3"] * Acc  # Eq. 50
                E = -a["a1"] * dU + a["a4"] * V + a["a5"] * Acc
                K_ef = K_t + a["a0"] * M + a["a1"] * D  # 4.5, Eq. 48
                F_ef = R_ext + M @ G + D @ E  # 4.6, Eq. 49
            else:
                K_ef, F_ef = K_t, R_ext

            residual = F_ef - F_int  # Eq. 37
            R_norm = np.linalg.norm(residual[free])
            if it == 1:
                R1_norm = R_norm

            # 4.7 solve K_ef du = residual (Eq. 51) for the free DOFs; the prescribed
            # DOFs move straight to their target in the first iteration.
            du = np.zeros(n)
            du[fixed] = U_target[fixed] - U[fixed]
            rhs = residual[free] - K_ef[free][:, fixed] @ du[fixed]
            du[free] = spla.spsolve(K_ef[free][:, free].tocsc(), rhs)

            # 4.8 update displacements (Eq. 52)
            U = U + du
            dU = dU + du

            # 4.9 convergence (Eqs. 34-35); fmin ignores 0/0 at the first iteration
            with np.errstate(divide="ignore", invalid="ignore"):
                disp_err = np.fmin(np.linalg.norm(du), np.linalg.norm(du) / np.linalg.norm(dU))
                force_err = np.fmin(R_norm, R_norm / R1_norm)
            if on_iteration:
                on_iteration(Iteration(step, it, residual, F_int, du, disp_err, force_err))
            if disp_err < settings.tol_disp and force_err < settings.tol_res:
                converged = True
                break

        if not converged:
            result.finish(converged=False, message=f"no convergence at step {step}",
                          elapsed=clock.perf_counter() - start)
            return result

        # 5. velocities and accelerations (Eqs. 32-33)
        if dynamic:
            Acc_old = Acc
            Acc = a["a0"] * dU - a["a2"] * V - a["a3"] * Acc_old
            V = V + a["a6"] * Acc_old + a["a7"] * Acc

        if corrections.update_geometry_after_convergence:
            beams.update(du)

        result.store(step, t, info.static_ratio, it, U, beams.forces)
        if on_step:
            on_step(step, n_steps, it)

    result.finish(converged=True, message="analysis finished",
                  elapsed=clock.perf_counter() - start)
    return result
