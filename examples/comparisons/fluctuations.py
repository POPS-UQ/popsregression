"""Loss-fluctuation diagnostics for the moment condition of the PAC-Bayes bound.

For a fixed hyperparameter ``Psi`` with predictive density ``p_Psi`` and
per-datum log loss ``l = -log p_Psi(Y | X)``, the one-sided cumulant
generating function entering Theorem 1 (``Psi_0H``) and Corollary 1
(``subgamma-H``) of the paper is

    psi(t) = log E exp{t [G(Psi) - l]} = t G + log E[p^t],   0 < t <= 1.

Because ``u -> u^t`` is concave, ``E[p^t] <= (E p)^t``, so

    psi(t) <= t J(Psi),   J(Psi) = log E p_Psi - E log p_Psi >= 0,

with equality at ``t = 1``. ``J`` is the Kullback--Leibler divergence of the
data law from its reweighting by the predictive density (``dQ ∝ p dP``); it
is zero for a predictive whose density at the data is constant, about
``(1 - log 2) / 2 = 0.153`` nats for a calibrated fixed-width Gaussian, and
infinite exactly when the predictive is not population-admissible. For i.i.d.
data and ``lambda = N`` the moment term of Theorem 1 is therefore exactly
``Psi_0H(N, N) = log E_{pi_0H} exp{N J(Psi)}``.

For smaller ``t`` the one-sided Bennett inequality gives the sub-gamma form of
Corollary 1 with ``s^2 = Var(l)`` and ``c = (G - a_Psi) / 3``, valid for
``0 < t < 1 / c``, where ``a_Psi`` is a lower bound on the loss over the whole
input and output domain. It requires an upper bound on the density but no
bound on the upper tail of the loss. For the POPS ellipse family the density
bound is analytic: every half-width is at least the offset radius ``delta``
(``min_half_width`` for the PAC path), so ``a_Psi`` follows from
:func:`domain_minimum_loss` without looking at data. The smallest loss seen
on a finite test set is only a plug-in estimate of it.

The floor-contaminated loss lies in ``[a, b]`` with ``b = log(R_y / beta)``,
so Hoeffding's lemma gives the uniform sub-Gaussian bound
``psi(t) <= t^2 (b - a)^2 / 8`` for every ``t`` and every ``Psi``
(:func:`hoeffding_cgf`); it is a population guarantee for the floored loss
only and certifies nothing about the unfloored loss.

The helpers below estimate these quantities from held-out losses. Plug-in
estimates (``cgf``, ``jensen_gap``, sample variances, test-set minima) are
descriptive: they are not substitutes for the constants of a certified
bound, and no finite sample establishes a condition uniformly over a
hyperprior.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

from popsregression._projected_ball import (
    floor_contaminated_loss_bounds,
    log_norm_constant,
    projected_ball_logpdf,
)


def floored_losses(mean, half, y, n_dim, beta, y_bounds):
    """Floor-contaminated log losses and the outside-support mask.

    ``mean``, ``half`` have shape (m, n) (one row per hyperparameter draw);
    ``y`` has shape (n,).
    """
    log_ball = projected_ball_logpdf(y[None, :] - mean, half, n_dim)
    log_floor = np.log(beta) - np.log(y_bounds[1] - y_bounds[0])
    loss = -np.logaddexp(np.log1p(-beta) + log_ball, log_floor)
    return loss, np.isneginf(log_ball)


def cgf(losses, t):
    """Plug-in ``psi(t) = t mean(l) + log mean exp(-t l)`` along the last axis."""
    losses = np.asarray(losses, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = losses.shape[-1]
    out = [
        tt * losses.mean(axis=-1) + logsumexp(-tt * losses, axis=-1) - np.log(n)
        for tt in t
    ]
    return np.stack(out, axis=-1)


def jensen_gap(losses):
    """Plug-in ``J = log mean(p) - mean(log p)``, i.e. ``psi(1)``."""
    return cgf(losses, 1.0)[..., 0]


def bennett_cgf(t, variance, gap_to_minimum):
    """Sub-gamma (Bernstein form of Bennett) bound ``v t^2 / (2 (1 - c t))``.

    ``gap_to_minimum = G - a_Psi``; returns ``inf`` where ``c t >= 1``.
    """
    t = np.asarray(t, dtype=float)
    c = gap_to_minimum / 3.0
    with np.errstate(divide="ignore"):
        return np.where(c * t < 1.0, variance * t * t / (2.0 * (1.0 - c * t)), np.inf)


def hoeffding_cgf(t, loss_range):
    """Uniform bound ``t^2 R^2 / 8`` for a loss confined to an interval of
    width ``R`` (Hoeffding's lemma)."""
    t = np.asarray(t, dtype=float)
    return t * t * loss_range**2 / 8.0


def domain_minimum_loss(ball_dim, beta, y_bounds, min_half_width):
    """Smallest floor-contaminated loss anywhere in the domain, from the
    analytic half-width floor (no data used)."""
    lower, _ = floor_contaminated_loss_bounds(
        ball_dim, beta=beta, y_bounds=y_bounds, min_half_width=min_half_width
    )
    return float(lower)


def bennett_range(gap_to_minimum):
    """``c = (G - a) / 3`` and the admissible range ``t < 1 / c``."""
    c = gap_to_minimum / 3.0
    return c, (np.inf if c <= 0 else 1.0 / c)


def soft_max(values, lam):
    """``(1 / lam) log mean exp(lam * values)``, the prior-averaged moment term."""
    values = np.asarray(values, dtype=float)
    return (logsumexp(lam * values) - np.log(values.size)) / lam


def calibrated_projected_ball_gap(n_dim):
    """``J`` of a projected-ball predictive whose standardized residuals follow
    the predictive itself (calibrated, fixed width); ``0`` for ``P = 1``
    (uniform) and ``-> (1 - log 2) / 2`` as ``P -> infinity``."""
    log_c = log_norm_constant(n_dim)
    k = 0.5 * (n_dim - 1)

    def logf(t):
        return log_c + k * np.log1p(-t * t)

    second = quad(lambda t: np.exp(2 * logf(t)), -1, 1)[0]
    entropy_term = quad(lambda t: np.exp(logf(t)) * logf(t), -1, 1)[0]
    return float(np.log(second) - entropy_term)


GAUSSIAN_CALIBRATED_GAP = 0.5 * (1.0 - np.log(2.0))
