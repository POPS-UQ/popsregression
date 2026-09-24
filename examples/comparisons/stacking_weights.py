"""
Predictive-density stacking on a cached log-density matrix.

Pure functions for the convex problem

    minimize_{q in simplex}  -sum_i w_i log( sum_k q_k exp(L[i, k]) ),

where ``L[i, k]`` is the cached log predictive density of a frozen
candidate ``k`` at held-out observation ``i`` and ``w`` are non-negative
row weights summing to one (equal weights, or equal *group* weights for
grouped observations). This is stacking of predictive densities in the
sense of Yao, Vehtari, Simpson and Gelman (2018), with the held-out
densities supplied by a pilot/certification split rather than by
leave-one-out or PSIS approximations. It is used by
:class:`comparisons.pops_dictionary.POPSPACCertificate` as the same-dictionary
comparison to the categorical Gibbs weights.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _row_weights(log_density, row_weights):
    n_rows = log_density.shape[0]
    if row_weights is None:
        return np.full(n_rows, 1.0 / n_rows)
    w = np.asarray(row_weights, dtype=float)
    if w.shape != (n_rows,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError(
            "row_weights must be non-negative, of shape (n_rows,), and sum to 1."
        )
    return w


def mixture_log_loss(weights, log_density, row_weights=None):
    """Weighted mixture negative log density and its exact gradient.

    Parameters
    ----------
    weights : ndarray of shape (n_candidates,)
        Categorical weights ``q`` (non-negative; zeros allowed).

    log_density : ndarray of shape (n_rows, n_candidates)
        Cached log densities ``log p_k(y_i | x_i)``; ``-inf`` entries are
        allowed as long as every row has at least one finite entry.

    row_weights : ndarray of shape (n_rows,), default=None
        Non-negative weights summing to one; uniform if None.

    Returns
    -------
    value : float
        ``-sum_i w_i log sum_k q_k exp(log_density[i, k])``.

    grad : ndarray of shape (n_candidates,)
        Gradient with respect to ``q``: ``-sum_i w_i p_ik / mix_i``.

    Examples
    --------
    >>> import numpy as np
    >>> L = np.log(np.array([[0.5, 0.1], [0.1, 0.5]]))
    >>> value, grad = mixture_log_loss(np.array([0.5, 0.5]), L)
    >>> bool(np.isclose(value, -np.log(0.3)))
    True
    """
    q = np.asarray(weights, dtype=float)
    L = np.asarray(log_density, dtype=float)
    w = _row_weights(L, row_weights)
    with np.errstate(divide="ignore"):
        log_q = np.log(q)
    log_mix = logsumexp(L + log_q, axis=1)
    value = -float(w @ log_mix)
    responsibility = np.exp(L - log_mix[:, None])
    grad = -(w @ responsibility)
    return value, grad


def _kkt_residual(q, grad):
    """KKT violation of ``q`` for the simplex-constrained mixture loss.

    At the optimum ``q^T grad = -1`` (the responsibilities average to
    one), so the multiplier is ``-1``: on the support ``grad_k = -1`` and
    off it ``grad_k >= -1``.
    """
    on = q > 0
    residual = 0.0
    if np.any(on):
        residual = float(np.max(np.abs(grad[on] + 1.0)))
    if np.any(~on):
        residual = max(residual, float(np.max(np.maximum(-1.0 - grad[~on], 0.0))))
    return residual


@dataclass(frozen=True)
class StackingSolution:
    """Result of :func:`solve_stacking_weights`.

    Attributes
    ----------
    weights : ndarray of shape (n_candidates,)
        Stacking weights on the simplex (exact zeros allowed).

    objective : float
        Mixture log loss at ``weights``.

    status : dict
        Solver provenance: ``method`` (``'slsqp'`` or ``'em_fallback'``),
        ``success``, ``message``, ``n_iter``, ``simplex_residual``
        (``|sum(q) - 1|`` and the largest negative entry before
        projection) and ``kkt_residual``. A failed solve is reported here,
        not disguised as an optimum.
    """

    weights: np.ndarray
    objective: float
    status: dict


def _project_simplex(q, zero_threshold):
    """Clip, zero negligible entries and renormalize."""
    q = np.where(q > zero_threshold, q, 0.0)
    total = q.sum()
    if total <= 0.0:
        raise ValueError("All stacking weights collapsed to zero.")
    return q / total


def solve_stacking_weights(
    log_density,
    row_weights=None,
    *,
    tol=1e-12,
    kkt_tol=1e-6,
    zero_threshold=1e-12,
    max_iter=1000,
    em_max_iter=20000,
):
    """Minimize the mixture log loss over the probability simplex.

    The problem is convex (its Hessian is a weighted second moment of the
    responsibilities), so any KKT point is a global optimum; the objective
    value is unique even when the weights are not (duplicated candidates).
    SLSQP with the analytic gradient is tried first and its KKT residual
    verified; if it fails, the multiplicative EM update
    ``q_k <- q_k sum_i w_i p_ik / mix_i`` (which preserves the simplex
    and never increases the loss) is run and reported as the fallback.

    Parameters
    ----------
    log_density : ndarray of shape (n_rows, n_candidates)
        Cached held-out log densities.

    row_weights : ndarray of shape (n_rows,), default=None
        Non-negative weights summing to one; uniform if None.

    tol : float, default=1e-12
        Objective tolerance of SLSQP.

    kkt_tol : float, default=1e-6
        Largest accepted KKT residual of the returned weights.

    zero_threshold : float, default=1e-12
        Weights at or below this value are set to exactly zero before the
        KKT check (``0 log 0 = 0`` in the categorical KL).

    max_iter : int, default=1000
        Maximum SLSQP iterations.

    em_max_iter : int, default=20000
        Maximum EM iterations of the fallback.

    Returns
    -------
    StackingSolution
        Weights, objective and solver status.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> L = np.log(rng.uniform(0.1, 1.0, size=(50, 3)))
    >>> sol = solve_stacking_weights(L)
    >>> bool(np.isclose(sol.weights.sum(), 1.0)) and bool(np.all(sol.weights >= 0))
    True
    """
    L = np.asarray(log_density, dtype=float)
    if L.ndim != 2 or L.shape[0] == 0 or L.shape[1] == 0:
        raise ValueError("log_density must be a non-empty 2d array.")
    if not np.all(np.isfinite(np.max(L, axis=1))):
        raise ValueError("Every row of log_density needs a finite entry.")
    w = _row_weights(L, row_weights)
    n_candidates = L.shape[1]
    q_init = np.full(n_candidates, 1.0 / n_candidates)

    def objective(q):
        return mixture_log_loss(q, L, w)

    res = minimize(
        objective,
        q_init,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_candidates,
        constraints=[{"type": "eq", "fun": lambda q: q.sum() - 1.0}],
        options={"ftol": tol, "maxiter": max_iter},
    )
    raw = np.asarray(res.x, dtype=float)
    simplex_residual = max(abs(float(raw.sum()) - 1.0), float(max(0.0, -raw.min())))
    q = _project_simplex(raw, zero_threshold)
    value, grad = mixture_log_loss(q, L, w)
    kkt = _kkt_residual(q, grad)
    status = {
        "method": "slsqp",
        "success": bool(res.success) and kkt <= kkt_tol,
        "message": str(res.message),
        "n_iter": int(res.nit),
        "simplex_residual": simplex_residual,
        "kkt_residual": kkt,
    }
    if status["success"]:
        return StackingSolution(q, value, status)

    # Fallback: monotone EM fixed point from the uniform start.
    q = q_init.copy()
    value, grad = mixture_log_loss(q, L, w)
    n_iter = 0
    kkt = _kkt_residual(q, grad)
    while kkt > kkt_tol and n_iter < em_max_iter:
        q = _project_simplex(q * (-grad), zero_threshold)
        value, grad = mixture_log_loss(q, L, w)
        kkt = _kkt_residual(q, grad)
        n_iter += 1
    status = {
        "method": "em_fallback",
        "success": kkt <= kkt_tol,
        "message": (
            f"SLSQP failed ({res.message}; kkt={status['kkt_residual']:.2e}); "
            "EM multiplicative updates used instead."
        ),
        "n_iter": n_iter,
        "simplex_residual": abs(float(q.sum()) - 1.0),
        "kkt_residual": kkt,
    }
    return StackingSolution(q, value, status)
