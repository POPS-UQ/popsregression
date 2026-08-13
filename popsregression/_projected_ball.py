"""
Projected-ball pushforward kernels.

Pure functions for the scalar pushforward of a uniform distribution on a
``P``-dimensional ellipsoid under a linear map ``theta -> phi @ theta``.
If ``theta`` is uniform on the ellipsoid ``(theta - mu)^T B^{-1} (theta - mu)
<= 1`` then ``t = phi @ theta`` has the projected-ball density

    p(t) = C_P / a * (1 - (t - m)^2 / a^2)^k,   |t - m| < a,

with ``m = phi @ mu``, ``a^2 = phi^T B phi``, ``k = (P - 1) / 2`` and
normalization ``C_P = Gamma(P/2 + 1) / (sqrt(pi) * Gamma((P + 1)/2))``.

Also provides the smooth continuation ``L_rho`` of ``log`` used as a
log-barrier in :class:`~popsregression.POPSRegressionEllipse`.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.special import gammaln


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
