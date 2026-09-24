"""
Projected-ball pushforward kernels.

Pure functions for the scalar pushforward of a uniform distribution on a
``P``-dimensional ellipsoid under a linear map ``theta -> phi @ theta``.
If ``theta`` is uniform on the ellipsoid ``(theta - mu)^T B^{-1} (theta - mu)
<= 1`` then ``t = phi @ theta`` has the projected-ball density

    p(t) = C_P / a * (1 - (t - m)^2 / a^2)^k,   |t - m| < a,

with ``m = phi @ mu``, ``a^2 = phi^T B phi``, ``k = (P - 1) / 2`` and
normalization ``C_P = Gamma(P/2 + 1) / (sqrt(pi) * Gamma((P + 1)/2))``.

Also provides the smooth continuation ``L_rho`` of ``log`` used as the
log-barrier of the ellipsoid fit, and the floor-contaminated density of the
PAC construction (``POPSEllipseRegression(regularization='PAC')``), which
mixes the exact projected-ball density with a uniform density on a known
bounded output interval so that its log loss is bounded.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.special import betainc, gammaln


def log_norm_constant(n_dim):
    """Log normalization constant ``log(C_P)`` of the projected-ball density.

    Computed with :func:`scipy.special.gammaln` so it neither under- nor
    overflows for large ``n_dim``.

    Parameters
    ----------
    n_dim : int
        Dimension ``P`` of the ball being projected.

    Returns
    -------
    float
        ``log(C_P)`` with ``C_P = Gamma(P/2 + 1) / (sqrt(pi) *
        Gamma((P + 1)/2))``.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(np.exp(log_norm_constant(1)), 0.5))
    True
    >>> bool(np.isclose(np.exp(log_norm_constant(2)), 2.0 / np.pi))
    True
    """
    half = 0.5 * n_dim
    return gammaln(half + 1.0) - 0.5 * np.log(np.pi) - gammaln(half + 0.5)


def smooth_log(q, rho):
    """Smooth continuation ``L_rho`` of ``log`` and its first two derivatives.

    For ``q >= rho`` this is exactly ``log(q)``; for ``q < rho`` the log is
    continued by its second-order Taylor expansion around ``rho`` so that the
    function is finite and (at least) C^2 for all real ``q``:

    .. math::

        L_\\rho(q) = \\log\\rho + (q-\\rho)/\\rho - (q-\\rho)^2 / (2\\rho^2),
        \\qquad q < \\rho.

    As ``rho -> 0``, ``L_rho(q) -> log(q)`` pointwise for ``q > 0``.

    Parameters
    ----------
    q : ndarray or float
        Argument(s); may be negative (the continued branch is used there).

    rho : float
        Positive continuation threshold.

    Returns
    -------
    value : ndarray
        ``L_rho(q)``.

    grad : ndarray
        ``L_rho'(q)`` (``1/q`` above the threshold, linear below).

    hess : ndarray
        ``L_rho''(q)`` (``-1/q^2`` above the threshold, ``-1/rho^2`` below).

    Examples
    --------
    >>> import numpy as np
    >>> val, grad, hess = smooth_log(np.array([0.5, -2.0]), 1e-2)
    >>> bool(np.isclose(val[0], np.log(0.5)))
    True
    >>> bool(np.isfinite(val[1]))
    True
    """
    q = np.asarray(q, dtype=float)
    above = q >= rho
    q_safe = np.where(above, q, 1.0)
    t = q - rho
    value = np.where(
        above,
        np.log(q_safe),
        np.log(rho) + t / rho - t * t / (2.0 * rho * rho),
    )
    grad = np.where(above, 1.0 / q_safe, (1.0 - t / rho) / rho)
    hess = np.where(above, -1.0 / (q_safe * q_safe), -1.0 / (rho * rho))
    return value, grad, hess


def projected_ball_logpdf(t, half_width, n_dim):
    """Log density of the projected-ball distribution.

    Parameters
    ----------
    t : ndarray or float
        Evaluation point(s), measured from the center of the distribution.

    half_width : float
        Support half-width ``a > 0``; the density is supported on
        ``(-a, a)``.

    n_dim : int
        Dimension ``P`` of the ball being projected.

    Returns
    -------
    ndarray
        ``log p(t)``; ``-inf`` outside the support.

    Examples
    --------
    >>> import numpy as np
    >>> logp = projected_ball_logpdf(np.array([0.0, 2.0]), 1.0, 3)
    >>> bool(np.isclose(np.exp(logp[0]), 0.75))
    True
    >>> bool(np.isneginf(logp[1]))
    True
    """
    t = np.asarray(t, dtype=float)
    k = 0.5 * (n_dim - 1)
    u = 1.0 - (t / half_width) ** 2
    inside = u > 0.0
    u_safe = np.where(inside, u, 1.0)
    logp = log_norm_constant(n_dim) - np.log(half_width) + k * np.log(u_safe)
    return np.where(inside, logp, -np.inf)


def projected_ball_pdf(t, half_width, n_dim):
    """Density of the projected-ball distribution (see
    :func:`projected_ball_logpdf`).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import quad
    >>> total = quad(lambda t: projected_ball_pdf(t, 2.0, 5), -2, 2)[0]
    >>> bool(np.isclose(total, 1.0))
    True
    """
    return np.exp(projected_ball_logpdf(t, half_width, n_dim))


def projected_ball_variance(half_width, n_dim):
    """Variance of the projected-ball distribution.

    A uniform draw from a ``P``-ball of radius 1 has per-coordinate variance
    ``1 / (P + 2)``, so the pushforward with support half-width ``a`` has
    variance ``a^2 / (P + 2)``.

    Parameters
    ----------
    half_width : float or ndarray
        Support half-width ``a``.

    n_dim : int
        Dimension ``P`` of the ball being projected.

    Returns
    -------
    float or ndarray
        ``half_width**2 / (n_dim + 2)``.

    Examples
    --------
    >>> bool(abs(projected_ball_variance(2.0, 2) - 1.0) < 1e-12)
    True
    """
    return half_width**2 / (n_dim + 2.0)


def projected_ball_cdf(t, half_width, n_dim):
    """Cumulative distribution function of the projected-ball distribution.

    ``(t / a + 1) / 2`` is ``Beta((P + 1)/2, (P + 1)/2)`` distributed, so
    the CDF is a regularized incomplete beta function; it is 0 below
    ``-a`` and 1 above ``a``.

    Parameters
    ----------
    t : ndarray or float
        Evaluation point(s), measured from the center of the distribution.

    half_width : float or ndarray
        Support half-width ``a > 0``; broadcast against ``t``.

    n_dim : int
        Dimension ``P`` of the ball being projected.

    Returns
    -------
    ndarray
        ``P(T <= t)``.

    Examples
    --------
    >>> import numpy as np
    >>> cdf = projected_ball_cdf(np.array([-2.0, 0.0, 2.0]), 1.0, 3)
    >>> bool(np.allclose(cdf, [0.0, 0.5, 1.0]))
    True
    """
    t = np.asarray(t, dtype=float)
    u = np.clip(0.5 * (t / half_width + 1.0), 0.0, 1.0)
    a = 0.5 * (n_dim + 1)
    return betainc(a, a, u)


def floor_contaminated_logpdf(
    residual, half_width, n_dim, *, beta, y, y_bounds, strict=True
):
    """Log of the floor-contaminated projected-ball density.

    On the known output interval ``Y0 = [y_lower, y_upper]`` of width
    ``R_y``, the certified predictive density of a candidate ellipsoid is

    .. math::

        p_\\beta(y \\mid x) = (1 - \\beta)\\, p(y \\mid x)
        + \\frac{\\beta}{R_y} \\mathbf 1[y \\in Y_0],

    with ``p`` the exact projected-ball density (support half-width
    ``half_width``). It is a proper density on ``R`` and its negative log is
    bounded on ``Y0`` (see :func:`floor_contaminated_loss_bounds`). The two
    terms are combined with :func:`numpy.logaddexp`, so an observation
    outside the ellipsoid support (where ``log p = -inf``) receives the
    finite floor loss ``log(R_y / beta)``.

    Parameters
    ----------
    residual : ndarray or float
        ``y - m(x)``, the observation measured from the candidate mean.

    half_width : ndarray or float
        Support half-width ``a(x) > 0`` of the projected-ball component;
        broadcast against ``residual``.

    n_dim : int
        Dimension ``P`` of the ball being projected.

    beta : float
        Floor weight in ``(0, 1)``.

    y : ndarray or float
        The observation itself; every entry must lie in ``y_bounds``.

    y_bounds : tuple of float
        ``(y_lower, y_upper)`` with ``y_lower < y_upper``.

    strict : bool, default=True
        If True (the certified path), any ``y`` outside ``y_bounds``
        raises. If False the density is evaluated on all of ``R``, where
        only the projected-ball term survives outside the interval; this
        is what normalization checks and predictive CDFs need.

    Returns
    -------
    ndarray
        ``log p_beta(y | x)``, finite for every ``y`` in ``y_bounds``.

    Raises
    ------
    ValueError
        If ``strict`` and any ``y`` lies outside ``y_bounds``: the certified
        protocol requires the declared interval to contain the population
        support, so such an observation is a protocol violation, never
        clipped.

    Examples
    --------
    >>> import numpy as np
    >>> logp = floor_contaminated_logpdf(
    ...     np.array([0.0, 5.0]), 1.0, 3, beta=0.1, y=np.array([0.0, 5.0]),
    ...     y_bounds=(-10.0, 10.0),
    ... )
    >>> bool(np.isclose(logp[1], np.log(0.1 / 20.0)))
    True
    """
    y_lower, y_upper = _validate_output_interval(y_bounds)
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must lie in (0, 1), got {beta!r}.")
    y = np.asarray(y, dtype=float)
    inside = (y >= y_lower) & (y <= y_upper)
    if strict and not np.all(inside):
        raise ValueError(
            "An observation lies outside the declared output interval "
            f"{(y_lower, y_upper)}; the certified protocol requires the "
            "interval to contain the population support and never clips."
        )
    log_ball = np.log1p(-beta) + projected_ball_logpdf(residual, half_width, n_dim)
    log_floor = np.log(beta) - np.log(y_upper - y_lower)
    log_floor = np.where(inside, log_floor, -np.inf)
    return np.logaddexp(log_ball, log_floor)


def floor_contaminated_loss_bounds(n_dim, *, beta, y_bounds, min_half_width):
    """Uniform bounds of the floor-contaminated log loss on the output interval.

    With projected-ball normalization ``C_P`` and a verified support
    half-width floor ``a(x) >= min_half_width``, the density satisfies
    ``beta / R_y <= p_beta <= (1 - beta) C_P / min_half_width + beta / R_y``
    for every ``y`` in the interval, so its negative log lies in
    ``[loss_lower, loss_upper]`` with

    .. math::

        \\text{loss\\_lower} = -\\log\\big((1-\\beta) C_P / a_{\\min}
        + \\beta / R_y\\big), \\qquad
        \\text{loss\\_upper} = \\log(R_y / \\beta).

    Parameters
    ----------
    n_dim : int
        Dimension ``P`` of the ball being projected.

    beta : float
        Floor weight in ``(0, 1)``.

    y_bounds : tuple of float
        ``(y_lower, y_upper)``.

    min_half_width : float
        Positive lower bound on the support half-width over the declared
        input domain.

    Returns
    -------
    loss_lower : float
        Lower bound of the loss.

    loss_upper : float
        Upper bound of the loss.

    Examples
    --------
    >>> lo, hi = floor_contaminated_loss_bounds(
    ...     3, beta=0.1, y_bounds=(-1.0, 1.0), min_half_width=0.5
    ... )
    >>> bool(lo < hi)
    True
    """
    y_lower, y_upper = _validate_output_interval(y_bounds)
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must lie in (0, 1), got {beta!r}.")
    if not min_half_width > 0.0:
        raise ValueError("min_half_width must be positive.")
    width = y_upper - y_lower
    peak = (1.0 - beta) * np.exp(log_norm_constant(n_dim)) / min_half_width
    loss_lower = -np.log(peak + beta / width)
    loss_upper = np.log(width / beta)
    return float(loss_lower), float(loss_upper)


def _validate_output_interval(y_bounds):
    """Check and unpack ``(y_lower, y_upper)``."""
    try:
        y_lower, y_upper = (float(v) for v in y_bounds)
    except (TypeError, ValueError) as err:
        raise ValueError("y_bounds must be a pair (y_lower, y_upper).") from err
    if not (np.isfinite(y_lower) and np.isfinite(y_upper)) or y_lower >= y_upper:
        raise ValueError(
            "y_bounds must be a finite interval with y_lower < y_upper, "
            f"got {(y_lower, y_upper)}."
        )
    return y_lower, y_upper
