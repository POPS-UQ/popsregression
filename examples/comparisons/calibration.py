"""Calibration curves and areas shared by the paper studies.

Two different questions are answered here, and they should not be confused:

- **Pooled error-distribution agreement** (:func:`error_pp_curve`, the ACE
  figure metric). Every test error ``|y_i - m_i|`` is pooled into one
  observed distribution, every predictive draw ``|y*_i - m_i|`` into one
  predicted distribution, and the two cumulative distribution functions are
  plotted against each other (a probability-probability, or P-P, curve).
  Agreement says that the *spread of errors over the whole test set* is
  right; it does not say that each input's interval is right.
- **Input-conditional calibration** (:func:`pit_curve`). Each test target is
  mapped through its own predictive cumulative distribution function,
  ``u_i = F_i(y_i)`` (the probability integral transform). If every
  predictive were correct for its own input the ``u_i`` would be uniform.

Both produce a curve ``C(u)`` on ``[0, 1]`` that equals ``u`` when calibrated,
and two areas:

- signed area ``A_s = int_0^1 (C(u) - u) du``;
- unsigned area ``A_abs = int_0^1 |C(u) - u| du``, the primary ranking
  (smaller is better).

``A_abs = 0`` if and only if ``C(u) = u`` everywhere. ``A_s = 0`` does NOT
imply agreement (positive and negative parts can cancel), and a positive
``A_s`` alone does NOT establish that one distribution is stochastically
smaller than the other (that needs ``C(u) >= u`` for every ``u``). For the
pooled error curve a positive ``A_s`` indicates net over-confidence (the
predicted errors are, on balance, smaller than the observed ones) and a
negative ``A_s`` net under-confidence.

All functions accept per-point ``weights`` (normalised internally); the
Burgers studies weight every simulator case equally.
"""

import numpy as np


def _normalised(weights, n):
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != n or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with one entry per point.")
    return w / w.sum()


def group_weights(groups):
    """Weights giving every group (e.g. simulator case) equal total mass."""
    if groups is None:
        return None
    _, inverse, counts = np.unique(
        np.asarray(groups), return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    return 1.0 / (counts.size * counts[inverse])


def step_curve_areas(knots, values):
    """Signed and unsigned area between a step curve and the parity line.

    ``C(u) = values[k]`` on ``(knots[k], knots[k + 1]]``, with ``knots``
    increasing from 0 to 1. Both integrals are exact.
    """
    a, b, c = knots[:-1], knots[1:], np.asarray(values, dtype=float)
    signed = np.sum(c * (b - a) - 0.5 * (b * b - a * a))
    below = c <= a  # |c - u| = u - c on the whole step
    above = c >= b
    inside = ~(below | above)
    unsigned = np.where(
        below,
        0.5 * (b * b - a * a) - c * (b - a),
        np.where(
            above,
            c * (b - a) - 0.5 * (b * b - a * a),
            0.5 * ((c - a) ** 2 + (b - c) ** 2) * inside,
        ),
    )
    return float(signed), float(np.sum(unsigned))


def pit_curve(pit, weights=None):
    """Input-conditional calibration curve from PIT values ``u_i = F_i(y_i)``.

    Returns ``(u, C, signed_area, unsigned_area)`` where ``C(u)`` is the
    weighted fraction of PIT values at or below ``u`` (a right-continuous
    step function, returned at its jump points with ``u[0] = 0`` and
    ``u[-1] = 1``).
    """
    pit = np.clip(np.asarray(pit, dtype=float).ravel(), 0.0, 1.0)
    w = _normalised(weights, pit.size)
    order = np.argsort(pit)
    u = np.concatenate([[0.0], pit[order], [1.0]])
    C = np.concatenate([[0.0], np.cumsum(w[order]), [1.0]])
    # On (u[k], u[k+1]] the curve equals C[k].
    signed, unsigned = step_curve_areas(u, C[:-1])
    return u, C, signed, unsigned


def error_pp_curve(observed_errors, predicted_errors, weights=None):
    """Pooled P-P curve of absolute errors, and its signed/unsigned areas.

    Parameters
    ----------
    observed_errors : array of shape (n,)
        Test errors ``y_i - m_i`` about each method's point prediction.
    predicted_errors : array of shape (n, s)
        ``s`` predictive draws of the same error at each test input
        (parameter-only, no residual noise).
    weights : array of shape (n,), default=None
        Weight of each test input (shared by its ``s`` draws).

    Returns
    -------
    u, C : ndarray
        The curve ``C(u) = F_pred(F_obs^{-1}(u))`` at the jumps of the
        observed distribution function (``u`` from 0 to 1).
    signed_area, unsigned_area : float
        ``int (C - u) du`` and ``int |C - u| du``, exact for these step
        functions.
    """
    obs = np.abs(np.asarray(observed_errors, dtype=float).ravel())
    pred = np.abs(np.asarray(predicted_errors, dtype=float))
    if pred.ndim != 2 or pred.shape[0] != obs.size:
        raise ValueError("predicted_errors must have shape (n_points, n_draws).")
    w = _normalised(weights, obs.size)
    order = np.argsort(obs)
    t = obs[order]
    u = np.concatenate([[0.0], np.cumsum(w[order])])
    u[-1] = 1.0
    flat = pred.ravel()
    flat_w = np.repeat(w / pred.shape[1], pred.shape[1])
    sort = np.argsort(flat)
    cum = np.concatenate([[0.0], np.cumsum(flat_w[sort])])
    # F_pred(t_k) for the k-th observed error; on (u[k-1], u[k]] the curve
    # C(u) = F_pred(F_obs^{-1}(u)) equals F_pred(t_k).
    C_steps = cum[np.searchsorted(flat[sort], t, side="right")]
    signed, unsigned = step_curve_areas(u, C_steps)
    return u, np.concatenate([[0.0], C_steps]), signed, unsigned


def central_interval(values, level=0.9):
    """Median and central ``level`` interval of finite ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    tail = 0.5 * (1.0 - level)
    lo, med, hi = np.quantile(v, [tail, 0.5, 1.0 - tail])
    return float(med), float(lo), float(hi)
