"""Deterministic viscous Burgers solver shared by the Burgers examples.

Solves ``u_t + u u_x = nu u_xx`` on the periodic domain ``[0, 2 pi)`` from
``u(x, 0) = A sin x`` on a 96-point grid, for parameters drawn uniformly from
the box ``nu in NU_RANGE``, ``A in AMP_RANGE``, ``t in T_RANGE``. Used by
``example_burgers_pod.py``, ``example_burgers_qoi.py`` and
``comparisons.harness``.
"""

import numpy as np
from scipy.integrate import solve_ivp

SEED = 1
NU_RANGE = (0.012, 0.08)
AMP_RANGE = (0.7, 1.3)
T_RANGE = (0.15, 0.95)
N_GRID = 96


def burgers_solution(nu, amplitude, t_final, n_grid=N_GRID):
    """Grid and solution ``u(x, t_final)`` by method of lines (RK45)."""
    x = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dx = x[1] - x[0]
    u0 = amplitude * np.sin(x)

    def rhs(_, u):
        flux = 0.5 * u**2
        flux_x = (np.roll(flux, -1) - np.roll(flux, 1)) / (2.0 * dx)
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2
        return -flux_x + nu * u_xx

    sol = solve_ivp(
        rhs, (0.0, t_final), u0, method="RK45", rtol=2e-6, atol=2e-8, max_step=0.01
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return x, sol.y[:, -1]


def scale(v, limits):
    """Map ``v`` from ``limits`` linearly onto ``[-1, 1]``."""
    lo, hi = limits
    return 2.0 * (np.asarray(v) - lo) / (hi - lo) - 1.0


def draw_cases(rng, n):
    """``n`` parameter triples ``(nu, A, t)`` uniform on the parameter box."""
    return np.column_stack(
        [
            rng.uniform(*NU_RANGE, n),
            rng.uniform(*AMP_RANGE, n),
            rng.uniform(*T_RANGE, n),
        ]
    )
