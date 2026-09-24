"""
Pilot-split finite-dictionary PAC certificate.

A bounded-loss PAC-Bayes certificate for the population log-risk of a
finite categorical mixture of uniform-ellipsoid predictive densities. It is
deliberately separate from the empirical-Bayes (EB) Laplace layer of the
one-sample ellipsoid estimator: every quantity on the right side of the
bound is an exact finite sum over a dictionary frozen by an independent
pilot sample, and a one-sample fit can never produce a certified result.

Protocol (conditional on the pilot sample ``D0``, over the draw of the
independent certification sample ``D1``):

1. ``fit_pilot(X_pilot, y_pilot)`` fits one ellipsoid on ``D0`` and freezes
   the centering/whitening, the center, the candidate scale grid, the
   categorical hyperprior ``q0``, the floor weight ``beta``, the declared
   output interval, the analytic half-width floor and the temperature grid.
2. ``certify(X_cert, y_cert)`` evaluates the frozen candidate densities on
   ``D1`` and returns the full right side of the bound for categorical
   Gibbs weights, fixed-prior weights, the best single candidate and
   held-out predictive-stacking weights, all on the same confidence event.

The certified quantity is the population risk of the floor-contaminated
predictive mixture on the declared output interval (``target =
'floor_contaminated_predictive_log_risk'``). It is not a coverage
guarantee for any interval.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp
from sklearn.utils import check_random_state

from popsregression._ellipse import _EllipsoidPosterior
from popsregression._projected_ball import (
    _validate_output_interval,
    floor_contaminated_logpdf,
    floor_contaminated_loss_bounds,
    projected_ball_cdf,
    projected_ball_logpdf,
)

from .stacking_weights import mixture_log_loss, solve_stacking_weights

CERTIFICATE_TARGET = "floor_contaminated_predictive_log_risk"

_WEIGHT_RULES = ("gibbs", "fixed_prior", "single_best", "stacking")


# --- Exact categorical algebra -------------------------------------------


def categorical_kl(weights, prior_weights):
    """``KL(q || q0)`` for categorical weights with ``0 log 0 = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(categorical_kl([1.0, 0.0], [0.5, 0.5]), np.log(2.0)))
    True
    """
    q = np.asarray(weights, dtype=float)
    q0 = np.asarray(prior_weights, dtype=float)
    on = q > 0.0
    return float(np.sum(q[on] * np.log(q[on] / q0[on])))


def gibbs_weights(prior_weights, candidate_empirical, temperature):
    """Categorical Gibbs weights ``q_k ∝ q0_k exp(-lambda g_k)``.

    Examples
    --------
    >>> import numpy as np
    >>> q = gibbs_weights([0.5, 0.5], [0.0, np.log(3.0)], 1.0)
    >>> bool(np.allclose(q, [0.75, 0.25]))
    True
    """
    q0 = np.asarray(prior_weights, dtype=float)
    g = np.asarray(candidate_empirical, dtype=float)
    logits = np.log(q0) - float(temperature) * g
    return np.exp(logits - logsumexp(logits))


def categorical_pac_bound(
    weights,
    candidate_empirical,
    prior_weights,
    temperature,
    *,
    n_units,
    loss_range,
    n_temperatures,
    failure_probability,
):
    """Right side of the finite-dictionary bounded-loss PAC-Bayes inequality.

    For any simplex vector ``q`` (the bound is simultaneous over ``q``),
    candidate empirical losses ``g_hat``, prior ``q0``, temperature
    ``lambda`` from a predeclared grid of size ``|Lambda|``, ``N1``
    independent units, loss range ``R`` and confidence ``xi``,

    .. math::

        \\text{raw\\_bound} = \\sum_k q_k \\hat g_k
        + \\frac{KL(q\\|q_0) + \\log(|\\Lambda| / \\xi)}{\\lambda}
        + \\frac{\\lambda R^2}{8 N_1}.

    Returns
    -------
    dict
        ``empirical``, ``kl``, ``complexity_penalty``,
        ``concentration_penalty`` and ``raw_bound``.

    Examples
    --------
    >>> parts = categorical_pac_bound(
    ...     [0.5, 0.5], [1.0, 2.0], [0.5, 0.5], 4.0, n_units=10,
    ...     loss_range=2.0, n_temperatures=1, failure_probability=0.05,
    ... )
    >>> bool(abs(parts["empirical"] - 1.5) < 1e-12)
    True
    """
    q = np.asarray(weights, dtype=float)
    g = np.asarray(candidate_empirical, dtype=float)
    lam = float(temperature)
    if q.shape != g.shape or np.any(q < 0.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("weights must be a probability vector matching g_hat.")
    if lam <= 0.0:
        raise ValueError("temperature must be positive.")
    empirical = float(q @ g)
    kl = categorical_kl(q, prior_weights)
    complexity = (kl + np.log(n_temperatures / failure_probability)) / lam
    concentration = lam * loss_range**2 / (8.0 * n_units)
    return {
        "empirical": empirical,
        "kl": kl,
        "complexity_penalty": float(complexity),
        "concentration_penalty": float(concentration),
        "raw_bound": float(empirical + complexity + concentration),
    }


# --- Frozen state and results --------------------------------------------


def _frozen(array, dtype=float):
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _digest(*items):
    """SHA-256 over a sequence of scalars, strings and arrays."""
    h = hashlib.sha256()
    for item in items:
        if isinstance(item, np.ndarray):
            h.update(str(item.dtype).encode())
            h.update(str(item.shape).encode())
            if item.dtype.kind == "O":
                h.update(repr(item.tolist()).encode())
            else:
                h.update(np.ascontiguousarray(item).tobytes())
        else:
            h.update(repr(item).encode())
        h.update(b"|")
    return h.hexdigest()


@dataclass(frozen=True)
class PilotState:
    """Everything frozen by the pilot sample.

    All arrays are read-only. ``digest()`` hashes the complete state so
    that a certification call can prove it left the pilot untouched.
    """

    n_features: int
    with_intercept: bool
    x_offset: np.ndarray
    y_offset: float
    whiten: np.ndarray
    center: np.ndarray
    U: np.ndarray
    baseline: str
    baseline_factor: np.ndarray
    baseline_ridge: float
    ball_dim: int
    B_pilot: np.ndarray
    constant_feature: object
    constant_value: float
    candidate_scales: np.ndarray
    prior_weights: np.ndarray
    beta: float
    y_bounds: tuple
    min_half_width: float
    lambdas: np.ndarray
    failure_probability: float
    loss_lower: float
    loss_upper: float
    floor_construction: str
    n_pilot_rows: int
    n_pilot_units: int
    pilot_sample_ids: np.ndarray
    pilot_group_ids: np.ndarray
    pilot_objective: float
    pilot_coverage_fraction: float

    def digest(self):
        """SHA-256 digest of the complete pilot state."""
        values = []
        for name in sorted(self.__dataclass_fields__):
            values.append(name)
            values.append(getattr(self, name))
        return _digest(*values)


@dataclass(frozen=True)
class WeightRuleCertificate:
    """The certificate evaluated at one categorical weight vector.

    Attributes
    ----------
    rule : str
        ``'gibbs'``, ``'fixed_prior'``, ``'single_best'`` or ``'stacking'``.

    weights : ndarray of shape (n_candidates,)
        The categorical weights.

    selected_lambda : float
        Temperature from the predeclared grid minimizing ``raw_bound``.

    empirical : float
        PAC empirical term ``sum_k q_k g_hat_k`` (average candidate loss).

    kl, complexity_penalty, concentration_penalty : float
        The remaining terms of the bound at ``selected_lambda``.

    raw_bound : float
        The complete PAC right side (the certified quantity).

    capped_bound : float
        ``min(raw_bound, loss_upper)``; the trivial cap is never a
        substitute for ``raw_bound``.

    is_nonvacuous : bool
        ``raw_bound < loss_upper``.

    mixture_empirical : float
        Empirical log loss of the mixture density (unit-averaged). NOT the
        PAC empirical term.

    jensen_gap : float
        ``empirical - mixture_empirical >= 0``.

    floor_fraction : float
        Mean fraction of the mixture predictive density supplied by the
        uniform floor at the certification observations.

    unit_mixture_losses : ndarray of shape (n_units,)
        Per-independent-unit mixture log loss.

    bounds_by_lambda : ndarray of shape (n_lambdas,)
        ``raw_bound`` at every temperature of the grid.

    weights_by_lambda : ndarray of shape (n_lambdas, n_candidates)
        The weights used at each temperature (constant except for Gibbs).

    solver_status : dict
        Provenance of the weights (stacking solver status; empty
        otherwise).
    """

    rule: str
    weights: np.ndarray
    selected_lambda: float
    empirical: float
    kl: float
    complexity_penalty: float
    concentration_penalty: float
    raw_bound: float
    capped_bound: float
    is_nonvacuous: bool
    mixture_empirical: float
    jensen_gap: float
    floor_fraction: float
    unit_mixture_losses: np.ndarray
    bounds_by_lambda: np.ndarray
    weights_by_lambda: np.ndarray
    solver_status: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PACCertificateResult:
    """Result of :meth:`POPSPACCertificate.certify`.

    The top-level bound attributes refer to the PAC-selected categorical
    Gibbs weights; ``rules`` holds the same certificate evaluated at the
    fixed-prior, single-best and predictive-stacking weights on the same
    dictionary and the same confidence event.

    Attributes
    ----------
    certified : bool
        True only when every protocol invariant passed: the pilot state is
        unchanged, the analytic half-width floor is established, every
        output lies in the declared interval and the bound is finite.

    certification_status : str
        Human-readable statement of the protocol and its assumptions.

    target : str
        ``'floor_contaminated_predictive_log_risk'``.

    raw_bound, capped_bound, is_nonvacuous, empirical, kl,
    complexity_penalty, concentration_penalty, posterior_weights,
    selected_lambda, mixture_empirical, jensen_gap : see
        :class:`WeightRuleCertificate` (Gibbs rule).

    prior_weights : ndarray of shape (n_candidates,)
        The frozen hyperprior ``q0``.

    candidate_empirical : ndarray of shape (n_candidates,)
        ``g_hat_k``, the unit-averaged candidate losses.

    unit_losses : ndarray of shape (n_units, n_candidates)
        Per-independent-unit candidate losses (within-group means).

    loss_lower, loss_upper, loss_range : float
        The analytic loss interval entering the Hoeffding term.

    n_independent_units, n_rows : int
        ``N1`` and the number of feature rows.

    independence_assumptions : dict
        What was verified from identifiers and what remains a caller
        assumption.

    rules : dict of WeightRuleCertificate
        Keyed by rule name.

    row_log_density : ndarray of shape (n_rows, n_candidates)
        Floor-contaminated candidate log densities at every row.

    row_log_density_exact : ndarray of shape (n_rows, n_candidates)
        Unmodified projected-ball log densities (``-inf`` outside support).

    fraction_outside_support : ndarray of shape (n_candidates,)
        Fraction of rows outside each candidate's compact support.

    floor_fraction : ndarray of shape (n_candidates,)
        Mean fraction of each candidate's contaminated density supplied by
        the uniform floor.

    loss_summary : dict
        Descriptive tail statistics of the contaminated unit losses and of
        the unmodified row losses (variance, upper quantiles, maximum,
        non-finite counts). Descriptive only.

    protocol : dict
        ``N0``, ``N1``, ``K``, ``beta``, ``y_bounds``, ``min_half_width``,
        ``xi``, ``lambdas``, ``candidate_scales`` and the pilot digest.
    """

    certified: bool
    certification_status: str
    target: str
    raw_bound: float
    capped_bound: float
    is_nonvacuous: bool
    empirical: float
    kl: float
    complexity_penalty: float
    concentration_penalty: float
    posterior_weights: np.ndarray
    prior_weights: np.ndarray
    selected_lambda: float
    mixture_empirical: float
    jensen_gap: float
    candidate_empirical: np.ndarray
    unit_losses: np.ndarray
    loss_lower: float
    loss_upper: float
    loss_range: float
    n_independent_units: int
    n_rows: int
    independence_assumptions: dict
    rules: dict
    row_log_density: np.ndarray
    row_log_density_exact: np.ndarray
    fraction_outside_support: np.ndarray
    floor_fraction: np.ndarray
    loss_summary: dict
    protocol: dict

    def bound_for_weights(self, weights, temperature=None):
        """Evaluate the same certificate at arbitrary simplex weights.

        Parameters
        ----------
        weights : array-like of shape (n_candidates,)
            Any probability vector.

        temperature : float, default=None
            A temperature of the predeclared grid. If None, the grid
            minimum of ``raw_bound`` is returned (still certified through
            the ``log(|Lambda| / xi)`` correction).

        Returns
        -------
        dict
            Bound components, plus ``temperature``, ``mixture_empirical``
            and ``jensen_gap``.
        """
        q = np.asarray(weights, dtype=float)
        lambdas = self.protocol["lambdas"]
        if temperature is None:
            candidates = [self.bound_for_weights(q, lam) for lam in lambdas]
            return min(candidates, key=lambda d: d["raw_bound"])
        lam = float(temperature)
        if not np.any(np.isclose(lambdas, lam, rtol=1e-12, atol=0.0)):
            raise ValueError("temperature must belong to the predeclared grid.")
        parts = categorical_pac_bound(
            q,
            self.candidate_empirical,
            self.prior_weights,
            lam,
            n_units=self.n_independent_units,
            loss_range=self.loss_range,
            n_temperatures=lambdas.size,
            failure_probability=self.protocol["xi"],
        )
        row_weights = self.protocol["row_weights"]
        mixture, _ = mixture_log_loss(q, self.row_log_density, row_weights)
        parts["temperature"] = lam
        parts["mixture_empirical"] = float(mixture)
        parts["jensen_gap"] = float(parts["empirical"] - mixture)
        return parts

    def summary(self):
        """Flat dictionary of scalar quantities for tables and CSV output."""
        out = {
            "certified": self.certified,
            "target": self.target,
            "n_rows": self.n_rows,
            "n_independent_units": self.n_independent_units,
            "n_candidates": self.prior_weights.size,
            "loss_lower": self.loss_lower,
            "loss_upper": self.loss_upper,
            "loss_range": self.loss_range,
        }
        for key in ("N0", "beta", "min_half_width", "xi", "n_lambdas"):
            out[key] = self.protocol[key]
        out["y_lower"], out["y_upper"] = self.protocol["y_bounds"]
        for name, rule in self.rules.items():
            for key in (
                "selected_lambda",
                "empirical",
                "kl",
                "complexity_penalty",
                "concentration_penalty",
                "raw_bound",
                "capped_bound",
                "is_nonvacuous",
                "mixture_empirical",
                "jensen_gap",
                "floor_fraction",
            ):
                out[f"{name}_{key}"] = getattr(rule, key)
        return out


# --- The certifier ---------------------------------------------------------


class POPSPACCertificate:
    """Pilot-split finite-dictionary PAC certificate for POPS ellipsoids.

    Fits one uniform-ellipsoid POPS posterior on an independent pilot
    sample, freezes a finite dictionary of candidate ellipsoids
    ``B_k = scale_k**2 * B_pilot + B_floor`` around it, and certifies the
    population log-risk of a categorical mixture of their
    floor-contaminated predictive densities on an independent
    certification sample. The bound (Corollary "pilot-split
    finite-dictionary certificate" of the accompanying paper) is

    .. math::

        \\mathbb E[-\\log \\bar p_{q,\\beta}(Y \\mid X)]
        \\le \\sum_k q_k \\hat G_k
        + \\frac{KL(q \\| q_0) + \\log(|\\Lambda|/\\xi)}{\\lambda}
        + \\frac{\\lambda R_\\ell^2}{8 N_1},

    simultaneously for every simplex vector ``q`` and every ``lambda`` of
    the predeclared grid, with probability at least ``1 - xi`` over the
    certification sample, conditional on the pilot. The guarantee concerns
    the floor-contaminated log-risk on the declared bounded output
    interval; it is not a coverage guarantee for any interval, nor a
    certificate for the unclipped compact-support loss, the smooth barrier
    objective, or the one-sample empirical-Bayes layer of
    :class:`~popsregression.POPSRegression`.

    Parameters
    ----------
    candidate_scales : array-like of shape (n_candidates,), \
            default=np.geomspace(0.5, 4.0, 17)
        Deterministic positive multipliers ``h_k`` of the pilot shape.

    prior_weights : array-like of shape (n_candidates,), default=None
        Strictly positive categorical hyperprior ``q0``; uniform if None.

    beta : float, default=0.02
        Weight of the uniform floor on the output interval.

    y_bounds : tuple of float
        Known population output support ``(y_lower, y_upper)``. This must
        come from a domain/simulator statement; sample extrema are not a
        support proof. Required.

    min_half_width : float, default=0.05
        Requested support half-width floor ``a_min``. It is enforced
        analytically by the protocol-fixed shape floor
        ``B_floor = a_min**2 e_c e_c^T`` on the constant coordinate ``c``
        of the feature map (see ``constant_feature``), giving
        ``a_k(x)**2 = h_k**2 s_pilot(x) + a_min**2 >= a_min**2`` for every
        candidate and every admissible input.

    lambdas : array-like of shape (n_lambdas,), \
            default=np.geomspace(0.25, 128.0, 24)
        Predeclared finite temperature grid ``Lambda``. Selecting the
        smallest bound over it costs ``log(|Lambda| / xi)``; a one-element
        grid reduces this to ``log(1 / xi)``.

    failure_probability : float, default=0.05
        ``xi``; the risk bound holds with probability at least ``1 - xi``.

    constant_feature : {'intercept'} or int, default='intercept'
        Where the analytic floor lives. ``'intercept'`` appends an
        intercept coordinate (features and targets are centered on the
        pilot; the whitened intercept coordinate is identically one). An
        integer names a column of ``X`` that must equal one nonzero
        constant on the pilot and on every later input, which is verified
        and otherwise rejected as outside the declared domain.

    pilot_options : dict, default=None
        Options of the pilot ellipsoid fit (the ``posterior_options`` keys
        of the ellipsoid engine behind ``POPSEllipseRegression``, e.g. ``rank``,
        ``baseline``, ``delta``); ``fit_intercept``, ``pac_bayes``,
        ``weights`` and ``random_state`` are controlled here.

    stacking : bool, default=True
        Also solve held-out predictive stacking on the frozen dictionary
        and certify its weights.

    random_state : int, RandomState instance or None, default=0
        Seed of the pilot ellipsoid initialization and of parameter draws.

    Attributes
    ----------
    pilot_ : PilotState
        The frozen pilot state, set by :meth:`fit_pilot`.

    result_ : PACCertificateResult
        The last certificate, set by :meth:`certify`.

    Examples
    --------
    >>> import numpy as np
    >>> from comparisons.pops_dictionary import POPSPACCertificate
    >>> rng = np.random.RandomState(0)
    >>> def data(n):
    ...     x = rng.uniform(-1, 1, n)
    ...     X = np.column_stack([np.ones(n), x, x**2])
    ...     return X, np.sin(3 * x)
    >>> X0, y0 = data(60)
    >>> X1, y1 = data(80)
    >>> certifier = POPSPACCertificate(
    ...     y_bounds=(-1.0, 1.0), min_half_width=0.05, constant_feature=0
    ... )
    >>> result = certifier.fit_pilot(X0, y0).certify(X1, y1)
    >>> bool(result.certified) and result.raw_bound > result.empirical
    True
    """

    def __init__(
        self,
        *,
        candidate_scales=None,
        prior_weights=None,
        beta=0.02,
        y_bounds=None,
        min_half_width=0.05,
        lambdas=None,
        failure_probability=0.05,
        constant_feature="intercept",
        pilot_options=None,
        stacking=True,
        random_state=0,
    ):
        self.candidate_scales = candidate_scales
        self.prior_weights = prior_weights
        self.beta = beta
        self.y_bounds = y_bounds
        self.min_half_width = min_half_width
        self.lambdas = lambdas
        self.failure_probability = failure_probability
        self.constant_feature = constant_feature
        self.pilot_options = pilot_options
        self.stacking = stacking
        self.random_state = random_state

    # -- protocol parameters -------------------------------------------

    def _validated_protocol(self):
        scales = (
            np.geomspace(0.5, 4.0, 17)
            if self.candidate_scales is None
            else np.atleast_1d(np.asarray(self.candidate_scales, dtype=float))
        )
        if scales.ndim != 1 or scales.size == 0 or np.any(scales <= 0.0):
            raise ValueError("candidate_scales must be a non-empty positive 1d array.")
        if self.prior_weights is None:
            q0 = np.full(scales.size, 1.0 / scales.size)
        else:
            q0 = np.asarray(self.prior_weights, dtype=float)
            if q0.shape != scales.shape or np.any(q0 <= 0.0):
                raise ValueError(
                    "prior_weights must be strictly positive with one entry per "
                    "candidate."
                )
            q0 = q0 / q0.sum()
        if not isinstance(self.beta, Real) or not 0.0 < self.beta < 1.0:
            raise ValueError("beta must lie in (0, 1).")
        if self.y_bounds is None:
            raise ValueError(
                "y_bounds is required: the certificate needs a known population "
                "output interval (not sample extrema)."
            )
        y_bounds = _validate_output_interval(self.y_bounds)
        if not isinstance(self.min_half_width, Real) or self.min_half_width <= 0:
            raise ValueError("min_half_width must be positive.")
        lambdas = (
            np.geomspace(0.25, 128.0, 24)
            if self.lambdas is None
            else np.atleast_1d(np.asarray(self.lambdas, dtype=float))
        )
        if lambdas.ndim != 1 or lambdas.size == 0 or np.any(lambdas <= 0.0):
            raise ValueError("lambdas must be a non-empty positive 1d array.")
        if np.unique(lambdas).size != lambdas.size:
            raise ValueError("lambdas must be distinct.")
        xi = self.failure_probability
        if not isinstance(xi, Real) or not 0.0 < xi < 1.0:
            raise ValueError("failure_probability must lie in (0, 1).")
        cf = self.constant_feature
        if not (cf == "intercept" or (isinstance(cf, Integral) and cf >= 0)):
            raise ValueError("constant_feature must be 'intercept' or a column index.")
        options = dict(self.pilot_options or {})
        for key in ("fit_intercept", "pac_bayes", "weights", "random_state"):
            if key in options:
                raise ValueError(f"{key!r} cannot be set through pilot_options.")
        return scales, q0, y_bounds, lambdas, float(xi), options

    # -- identifiers and independent units ------------------------------

    @staticmethod
    def _check_ids(name, ids, n_rows):
        if ids is None:
            return None
        ids = np.asarray(ids)
        if ids.shape != (n_rows,):
            raise ValueError(f"{name} must have shape (n_samples,).")
        return ids

    def _resolve_units(self, n_rows, sample_ids, group_ids, pilot=None):
        """Row-to-unit map, unit count and a record of what was verified."""
        sample_ids = self._check_ids("sample_ids", sample_ids, n_rows)
        group_ids = self._check_ids("group_ids", group_ids, n_rows)
        verified = []
        assumed = []
        if sample_ids is not None:
            if np.unique(sample_ids).size != n_rows:
                raise ValueError(
                    "Repeated sample_ids: reusing or resampling a record does not "
                    "create a new independent observation."
                )
            verified.append("distinct record identifiers")
            if pilot is not None and pilot.pilot_sample_ids.size:
                if np.intersect1d(sample_ids, pilot.pilot_sample_ids).size:
                    raise ValueError(
                        "sample_ids overlap with the pilot sample; pilot and "
                        "certification data must be disjoint."
                    )
                verified.append("no pilot/certification record overlap")
            elif pilot is not None:
                assumed.append("pilot records carried no identifiers")
        else:
            assumed.append("distinct rows are distinct records (no sample_ids)")
            if pilot is not None:
                assumed.append("no pilot/certification overlap (no sample_ids)")
        if group_ids is not None:
            if pilot is not None and pilot.pilot_group_ids.size:
                if np.intersect1d(group_ids, pilot.pilot_group_ids).size:
                    raise ValueError(
                        "group_ids overlap with the pilot sample; independent "
                        "cases must not be split across pilot and certification."
                    )
                verified.append("no pilot/certification group overlap")
            elif pilot is not None:
                assumed.append("pilot groups carried no identifiers")
            _, unit_index = np.unique(group_ids, return_inverse=True)
            unit_index = unit_index.ravel()
            verified.append("within-group losses averaged; N1 counts groups")
        else:
            unit_index = np.arange(n_rows)
            assumed.append("rows are independent units (no group_ids)")
        assumed.append(
            "independence of the units and agreement of the certification and "
            "deployment laws are caller assumptions, not software-verified facts"
        )
        n_units = int(unit_index.max()) + 1 if n_rows else 0
        record = {
            "sample_ids_provided": sample_ids is not None,
            "group_ids_provided": group_ids is not None,
            "verified": verified,
            "assumed": assumed,
        }
        return unit_index, n_units, record, sample_ids, group_ids

    # -- pilot ----------------------------------------------------------

    def fit_pilot(self, X, y, sample_ids=None, group_ids=None):
        """Fit the pilot ellipsoid and freeze the dictionary and protocol.

        Parameters
        ----------
        X : array-like of shape (n_pilot, n_features)
            Pilot inputs.

        y : array-like of shape (n_pilot,)
            Pilot outputs; must lie in ``y_bounds``.

        sample_ids : array-like of shape (n_pilot,), default=None
            Distinct record identifiers, used to reject overlap with the
            certification sample.

        group_ids : array-like of shape (n_pilot,), default=None
            Independent-case identifiers, likewise.

        Returns
        -------
        self : object
        """
        scales, q0, y_bounds, lambdas, xi, options = self._validated_protocol()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("X must be 2d with one row per entry of y.")
        if np.any(y < y_bounds[0]) or np.any(y > y_bounds[1]):
            raise ValueError(
                "Pilot outputs lie outside y_bounds, contradicting the declared "
                "population support."
            )
        n_rows, n_features = X.shape
        _, n_units, _, sample_ids, group_ids = self._resolve_units(
            n_rows, sample_ids, group_ids
        )

        use_intercept = self.constant_feature == "intercept"
        constant_value = 0.0
        if use_intercept:
            floor_construction = (
                "B_floor = min_half_width**2 on the appended intercept coordinate "
                "(identically 1 after pilot centering and whitening); "
                "a_k(x)**2 = h_k**2 s_pilot(x) + min_half_width**2 for every x."
            )
        else:
            j = int(self.constant_feature)
            if j >= n_features:
                raise ValueError("constant_feature index exceeds n_features.")
            column = X[:, j]
            if np.ptp(column) != 0.0 or column[0] == 0.0:
                raise ValueError(
                    f"Column {j} of the pilot X is not a nonzero constant, so the "
                    "analytic half-width floor cannot be established."
                )
            constant_value = float(column[0])
            floor_construction = (
                f"B_floor = (min_half_width / c)**2 e_{j} e_{j}^T with X[:, {j}] "
                f"== c = {constant_value!r} verified on the pilot and rejected "
                "otherwise on every later input; a_k(x)**2 = h_k**2 s_pilot(x) "
                "+ min_half_width**2 for every admissible x."
            )

        rng = check_random_state(self.random_state)
        ellipsoid = _EllipsoidPosterior(
            fit_intercept=use_intercept,
            random_state=int(rng.randint(np.iinfo(np.int32).max)),
            pac_bayes=False,
            **options,
        ).fit(X, y)

        if ellipsoid.baseline == "pops":
            baseline_factor = ellipsoid._baseline_factor
        else:
            baseline_factor = np.zeros((n_features + int(use_intercept), 0))
        loss_lower, loss_upper = floor_contaminated_loss_bounds(
            ellipsoid._ball_dim,
            beta=float(self.beta),
            y_bounds=y_bounds,
            min_half_width=float(self.min_half_width),
        )
        self.pilot_ = PilotState(
            n_features=int(n_features),
            with_intercept=bool(use_intercept),
            x_offset=_frozen(ellipsoid._x_offset),
            y_offset=float(ellipsoid._y_offset),
            whiten=_frozen(ellipsoid._whiten_W),
            center=_frozen(ellipsoid.center_whitened_),
            U=_frozen(ellipsoid.U_),
            baseline=str(ellipsoid.baseline),
            baseline_factor=_frozen(baseline_factor),
            baseline_ridge=float(ellipsoid.baseline_ridge),
            ball_dim=int(ellipsoid._ball_dim),
            B_pilot=_frozen(ellipsoid.ellipsoid_B_),
            constant_feature=self.constant_feature,
            constant_value=constant_value,
            candidate_scales=_frozen(scales),
            prior_weights=_frozen(q0),
            beta=float(self.beta),
            y_bounds=tuple(y_bounds),
            min_half_width=float(self.min_half_width),
            lambdas=_frozen(lambdas),
            failure_probability=xi,
            loss_lower=loss_lower,
            loss_upper=loss_upper,
            floor_construction=floor_construction,
            n_pilot_rows=int(n_rows),
            n_pilot_units=int(n_units),
            pilot_sample_ids=_frozen(
                np.empty(0) if sample_ids is None else sample_ids, dtype=None
            ),
            pilot_group_ids=_frozen(
                np.empty(0) if group_ids is None else group_ids, dtype=None
            ),
            pilot_objective=float(ellipsoid.objective_),
            pilot_coverage_fraction=float(ellipsoid.coverage_fraction_),
        )
        self._sqrt_cache = {}
        if hasattr(self, "result_"):
            del self.result_
        return self

    def _check_pilot(self):
        if not hasattr(self, "pilot_"):
            raise ValueError("Call fit_pilot before using the certificate.")
        return self.pilot_

    # -- frozen geometry ------------------------------------------------

    def _validate_inputs(self, X):
        pilot = self._check_pilot()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != pilot.n_features:
            raise ValueError(
                f"X must have shape (n_samples, {pilot.n_features}) as in the pilot."
            )
        if not pilot.with_intercept:
            j = int(pilot.constant_feature)
            if not np.all(X[:, j] == pilot.constant_value):
                raise ValueError(
                    f"Column {j} of X must equal the pilot constant "
                    f"{pilot.constant_value!r}: the input lies outside the "
                    "declared domain of the analytic half-width floor."
                )
        return X

    def _geometry(self, X):
        """Pilot mean, pilot squared widths and floor squared widths."""
        pilot = self._check_pilot()
        Xc = X - pilot.x_offset
        Z = Xc @ pilot.whiten
        if pilot.with_intercept:
            Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
        mean = Z @ pilot.center + pilot.y_offset
        s = np.sum((Z @ pilot.U) ** 2, axis=1)
        if pilot.baseline == "pops":
            D = (
                np.hstack([Xc, np.ones((Xc.shape[0], 1))])
                if pilot.with_intercept
                else Xc
            )
            DF = D @ pilot.baseline_factor
            s = s + np.einsum("ij,ij->i", DF, DF)
        elif pilot.baseline == "ridge":
            s = s + pilot.baseline_ridge * np.einsum("ij,ij->i", Z, Z)
        if pilot.with_intercept:
            s_floor = np.full(X.shape[0], pilot.min_half_width**2)
        else:
            ratio = X[:, int(pilot.constant_feature)] / pilot.constant_value
            s_floor = pilot.min_half_width**2 * ratio * ratio
        return mean, s, s_floor

    def candidate_half_widths(self, X):
        """Support half-widths ``a_k(x)`` of every candidate.

        Returns
        -------
        mean : ndarray of shape (n_samples,)
            Shared candidate mean ``m(x)``.

        half_widths : ndarray of shape (n_samples, n_candidates)
            ``sqrt(h_k**2 s_pilot(x) + s_floor(x)) >= min_half_width``.
        """
        X = self._validate_inputs(X)
        mean, s, s_floor = self._geometry(X)
        scales = self.pilot_.candidate_scales
        half = np.sqrt(scales[None, :] ** 2 * s[:, None] + s_floor[:, None])
        return mean, half

    def _log_densities(self, X, y):
        pilot = self._check_pilot()
        mean, half = self.candidate_half_widths(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        resid = (y - mean)[:, None]
        exact = projected_ball_logpdf(resid, half, pilot.ball_dim)
        contaminated = floor_contaminated_logpdf(
            resid,
            half,
            pilot.ball_dim,
            beta=pilot.beta,
            y=y[:, None],
            y_bounds=pilot.y_bounds,
        )
        return exact, contaminated

    def candidate_log_density(self, X, y):
        """Floor-contaminated log densities ``log p_{k,beta}(y | x)``.

        Returns
        -------
        ndarray of shape (n_samples, n_candidates)
        """
        X = self._validate_inputs(X)
        return self._log_densities(X, y)[1]

    # -- certification --------------------------------------------------

    def certify(self, X, y, sample_ids=None, group_ids=None):
        """Evaluate the certificate on an independent certification sample.

        Nothing pilot-dependent is refitted: the stored transforms and
        candidate ellipsoids are evaluated at ``X``, the losses are
        averaged within independent units, and the categorical weights of
        each rule are chosen. The pilot digest is checked before and after.

        Parameters
        ----------
        X : array-like of shape (n_cert, n_features)
            Certification inputs.

        y : array-like of shape (n_cert,)
            Certification outputs; an output outside ``y_bounds`` raises.

        sample_ids : array-like of shape (n_cert,), default=None
            Distinct record identifiers.

        group_ids : array-like of shape (n_cert,), default=None
            Independent-case identifiers; losses are averaged within a
            group and ``N1`` counts groups.

        Returns
        -------
        result : PACCertificateResult
        """
        pilot = self._check_pilot()
        digest_before = pilot.digest()
        X = self._validate_inputs(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_rows = X.shape[0]
        if n_rows == 0 or y.shape[0] != n_rows:
            raise ValueError("X and y must be non-empty with matching rows.")
        unit_index, n_units, record, _, _ = self._resolve_units(
            n_rows, sample_ids, group_ids, pilot=pilot
        )
        # floor_contaminated_logpdf raises on outputs outside y_bounds.
        exact, log_p = self._log_densities(X, y)
        loss = -log_p
        n_candidates = pilot.candidate_scales.size

        counts = np.bincount(unit_index, minlength=n_units).astype(float)
        row_weights = 1.0 / (n_units * counts[unit_index])
        unit_losses = np.zeros((n_units, n_candidates))
        np.add.at(unit_losses, unit_index, loss / counts[unit_index][:, None])
        g_hat = unit_losses.mean(axis=0)

        log_floor = np.log(pilot.beta) - np.log(pilot.y_bounds[1] - pilot.y_bounds[0])
        loss_range = pilot.loss_upper - pilot.loss_lower
        bound_kwargs = dict(
            n_units=n_units,
            loss_range=loss_range,
            n_temperatures=pilot.lambdas.size,
            failure_probability=pilot.failure_probability,
        )

        def certify_rule(rule, weights_by_lambda, solver_status=None):
            parts = [
                categorical_pac_bound(
                    q, g_hat, pilot.prior_weights, lam, **bound_kwargs
                )
                for q, lam in zip(weights_by_lambda, pilot.lambdas)
            ]
            bounds = np.array([p["raw_bound"] for p in parts])
            best = int(np.argmin(bounds))
            q = weights_by_lambda[best]
            with np.errstate(divide="ignore"):
                log_mix = logsumexp(log_p + np.log(q), axis=1)
            unit_mix = np.zeros(n_units)
            np.add.at(unit_mix, unit_index, -log_mix / counts[unit_index])
            mixture = float(unit_mix.mean())
            raw = parts[best]["raw_bound"]
            return WeightRuleCertificate(
                rule=rule,
                weights=_frozen(q),
                selected_lambda=float(pilot.lambdas[best]),
                empirical=parts[best]["empirical"],
                kl=parts[best]["kl"],
                complexity_penalty=parts[best]["complexity_penalty"],
                concentration_penalty=parts[best]["concentration_penalty"],
                raw_bound=raw,
                capped_bound=float(min(raw, pilot.loss_upper)),
                is_nonvacuous=bool(raw < pilot.loss_upper),
                mixture_empirical=mixture,
                jensen_gap=float(parts[best]["empirical"] - mixture),
                floor_fraction=float(np.mean(np.exp(log_floor - log_mix))),
                unit_mixture_losses=_frozen(unit_mix),
                bounds_by_lambda=_frozen(bounds),
                weights_by_lambda=_frozen(np.asarray(weights_by_lambda)),
                solver_status=dict(solver_status or {}),
            )

        rules = {}
        rules["gibbs"] = certify_rule(
            "gibbs",
            [gibbs_weights(pilot.prior_weights, g_hat, lam) for lam in pilot.lambdas],
        )
        rules["fixed_prior"] = certify_rule(
            "fixed_prior", [pilot.prior_weights] * pilot.lambdas.size
        )
        single = np.zeros(n_candidates)
        single[int(np.argmin(g_hat))] = 1.0
        rules["single_best"] = certify_rule(
            "single_best", [single] * pilot.lambdas.size
        )
        if self.stacking:
            solution = solve_stacking_weights(log_p, row_weights)
            rules["stacking"] = certify_rule(
                "stacking",
                [solution.weights] * pilot.lambdas.size,
                solver_status=solution.status,
            )

        gibbs = rules["gibbs"]
        finite = np.isfinite(gibbs.raw_bound)
        digest_after = pilot.digest()
        unchanged = digest_before == digest_after
        certified = bool(finite and unchanged)
        status = [
            (
                "pilot-split finite-dictionary PAC certificate: conditional on the "
                f"pilot sample, with probability >= {1 - pilot.failure_probability:g} "
                "over the certification sample, simultaneously for every categorical "
                f"weight vector and every temperature of the {pilot.lambdas.size}-"
                "point grid; target = floor-contaminated predictive log-risk on "
                f"y_bounds={pilot.y_bounds}"
            ),
            "floor: " + pilot.floor_construction,
            "verified: " + "; ".join(record["verified"]) if record["verified"] else "",
            "assumed: " + "; ".join(record["assumed"]),
        ]
        if not finite:
            status.append("NOT CERTIFIED: the bound is not finite")
        if not unchanged:
            status.append("NOT CERTIFIED: the pilot state changed during certify")

        with np.errstate(invalid="ignore"):
            exact_loss = -exact
            finite_mask = np.isfinite(exact_loss)
        loss_summary = {
            "contaminated_unit_loss": {
                "variance": unit_losses.var(axis=0, ddof=1 if n_units > 1 else 0),
                "q90": np.quantile(unit_losses, 0.9, axis=0),
                "q99": np.quantile(unit_losses, 0.99, axis=0),
                "max": unit_losses.max(axis=0),
            },
            "unmodified_row_loss": {
                "n_nonfinite": (~finite_mask).sum(axis=0),
                "max_finite": np.array(
                    [
                        (
                            exact_loss[finite_mask[:, k], k].max()
                            if finite_mask[:, k].any()
                            else np.nan
                        )
                        for k in range(n_candidates)
                    ]
                ),
                "q99_finite": np.array(
                    [
                        (
                            np.quantile(exact_loss[finite_mask[:, k], k], 0.99)
                            if finite_mask[:, k].any()
                            else np.nan
                        )
                        for k in range(n_candidates)
                    ]
                ),
                "variance_finite": np.array(
                    [
                        (
                            exact_loss[finite_mask[:, k], k].var()
                            if finite_mask[:, k].any()
                            else np.nan
                        )
                        for k in range(n_candidates)
                    ]
                ),
            },
        }
        protocol = {
            "N0": pilot.n_pilot_rows,
            "N0_units": pilot.n_pilot_units,
            "N1": n_units,
            "K": n_candidates,
            "beta": pilot.beta,
            "y_bounds": pilot.y_bounds,
            "min_half_width": pilot.min_half_width,
            "xi": pilot.failure_probability,
            "lambdas": pilot.lambdas,
            "n_lambdas": pilot.lambdas.size,
            "candidate_scales": pilot.candidate_scales,
            "ball_dim": pilot.ball_dim,
            "pilot_digest": digest_before,
            "row_weights": _frozen(row_weights),
            "unit_index": _frozen(unit_index, dtype=int),
        }
        result = PACCertificateResult(
            certified=certified,
            certification_status="\n".join(s for s in status if s),
            target=CERTIFICATE_TARGET,
            raw_bound=gibbs.raw_bound,
            capped_bound=gibbs.capped_bound,
            is_nonvacuous=gibbs.is_nonvacuous,
            empirical=gibbs.empirical,
            kl=gibbs.kl,
            complexity_penalty=gibbs.complexity_penalty,
            concentration_penalty=gibbs.concentration_penalty,
            posterior_weights=gibbs.weights,
            prior_weights=pilot.prior_weights,
            selected_lambda=gibbs.selected_lambda,
            mixture_empirical=gibbs.mixture_empirical,
            jensen_gap=gibbs.jensen_gap,
            candidate_empirical=_frozen(g_hat),
            unit_losses=_frozen(unit_losses),
            loss_lower=pilot.loss_lower,
            loss_upper=pilot.loss_upper,
            loss_range=float(loss_range),
            n_independent_units=int(n_units),
            n_rows=int(n_rows),
            independence_assumptions=record,
            rules=rules,
            row_log_density=_frozen(log_p),
            row_log_density_exact=_frozen(exact),
            fraction_outside_support=_frozen(np.mean(~finite_mask, axis=0)),
            floor_fraction=_frozen(np.mean(np.exp(log_floor - log_p), axis=0)),
            loss_summary=loss_summary,
            protocol=protocol,
        )
        self.result_ = result
        return result

    # -- predictive mixture --------------------------------------------

    def _weights(self, weights):
        pilot = self._check_pilot()
        if weights is None:
            if not hasattr(self, "result_"):
                raise ValueError("Pass weights or call certify first.")
            return self.result_.posterior_weights
        q = np.asarray(weights, dtype=float)
        if q.shape != pilot.candidate_scales.shape or np.any(q < 0):
            raise ValueError(
                "weights must be non-negative with one entry per candidate."
            )
        if not np.isclose(q.sum(), 1.0):
            raise ValueError("weights must sum to one.")
        return q

    def predict_logpdf(self, X, y, weights=None):
        """Log density of the floor-contaminated categorical mixture.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Outputs in ``y_bounds``.
        weights : array-like of shape (n_candidates,), default=None
            Categorical weights; the last certificate's Gibbs weights if
            None.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        q = self._weights(weights)
        log_p = self.candidate_log_density(X, y)
        with np.errstate(divide="ignore"):
            return logsumexp(log_p + np.log(q), axis=1)

    def predict(self, X, return_std=False, weights=None, contaminated=True):
        """Mean (and standard deviation) of the predictive mixture.

        With ``contaminated=True`` (the certified density) the moments
        include the uniform floor on the output interval, including its
        mean shift when the interval is not centered on the ellipsoid
        mean. With ``contaminated=False`` they refer to the uncontaminated
        parameter mixture ``sum_k q_k pi_k`` alone, the object to propagate
        through a simulator. Neither standard deviation is a certified
        interval; use :meth:`predict_interval` for exact quantiles.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
        y_std : ndarray of shape (n_samples,)
            Only if ``return_std=True``.
        """
        pilot = self._check_pilot()
        q = self._weights(weights)
        X = self._validate_inputs(X)
        mean, half = self.candidate_half_widths(X)
        comp_var = (half**2 / (pilot.ball_dim + 2.0)) @ q
        if not contaminated:
            return (mean, np.sqrt(comp_var)) if return_std else mean
        beta = pilot.beta
        y_lower, y_upper = pilot.y_bounds
        c0 = 0.5 * (y_lower + y_upper)
        u_var = (y_upper - y_lower) ** 2 / 12.0
        mixture_mean = (1.0 - beta) * mean + beta * c0
        second = (1.0 - beta) * (comp_var + mean**2) + beta * (u_var + c0**2)
        var = np.maximum(second - mixture_mean**2, 0.0)
        return (mixture_mean, np.sqrt(var)) if return_std else mixture_mean

    def predict_cdf(self, X, y, weights=None, contaminated=True):
        """CDF of the predictive mixture at ``y``.

        With ``contaminated=True`` this is the certified floor-contaminated
        density; with ``contaminated=False`` the pushforward of the
        uncontaminated parameter mixture ``sum_k q_k pi_k`` alone.
        """
        pilot = self._check_pilot()
        q = self._weights(weights)
        X = self._validate_inputs(X)
        y = np.asarray(y, dtype=float)
        mean, half = self.candidate_half_widths(X)
        resid = (y - mean)[:, None]
        ball = projected_ball_cdf(resid, half, pilot.ball_dim) @ q
        if not contaminated:
            return ball
        y_lower, y_upper = pilot.y_bounds
        floor = np.clip((y - y_lower) / (y_upper - y_lower), 0.0, 1.0)
        return (1.0 - pilot.beta) * ball + pilot.beta * floor

    def predict_interval(
        self, X, level=0.9545, weights=None, contaminated=True, n_bisect=80
    ):
        """Exact central interval of the predictive mixture.

        Quantiles ``(1 - level) / 2`` and ``(1 + level) / 2`` are found by
        vectorized bisection of :meth:`predict_cdf`, for the certified
        floor-contaminated density (``contaminated=True``) or for the
        uncontaminated parameter mixture (``contaminated=False``; note that
        the floor's uniform mass makes high-level contaminated intervals
        span most of the output interval). Either is a descriptive
        predictive interval of an explicitly identified density, not a
        PAC-certified coverage statement.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        pilot = self._check_pilot()
        q = self._weights(weights)
        X = self._validate_inputs(X)
        mean, half = self.candidate_half_widths(X)
        a_max = half[:, q > 0.0].max(axis=1)
        lo, hi = mean - a_max, mean + a_max
        if contaminated:
            y_lower, y_upper = pilot.y_bounds
            lo, hi = np.minimum(lo, y_lower), np.maximum(hi, y_upper)

        def quantile(prob):
            left, right = lo.copy(), hi.copy()
            for _ in range(n_bisect):
                mid = 0.5 * (left + right)
                below = self.predict_cdf(X, mid, q, contaminated) < prob
                left = np.where(below, mid, left)
                right = np.where(below, right, mid)
            return 0.5 * (left + right)

        return quantile(0.5 * (1.0 - level)), quantile(0.5 * (1.0 + level))

    def _candidate_sqrt(self, k):
        pilot = self._check_pilot()
        if k not in self._sqrt_cache:
            B = pilot.candidate_scales[k] ** 2 * pilot.B_pilot
            B = np.array(B)
            if pilot.with_intercept:
                B[-1, -1] += pilot.min_half_width**2
            else:
                j = int(pilot.constant_feature)
                B[j, j] += (pilot.min_half_width / pilot.constant_value) ** 2
            evals, evecs = eigh(B)
            self._sqrt_cache[k] = evecs * np.sqrt(np.maximum(evals, 0.0))
        return self._sqrt_cache[k]

    def sample_parameters(self, n_samples, weights=None, random_state=None):
        """Draw parameter vectors from the uncontaminated parameter mixture.

        Each draw selects one candidate from the categorical weights and
        then one parameter vector uniformly from that candidate ellipsoid
        (an affine map of a uniform unit-ball draw), so a single draw
        defines a complete surrogate field for propagation. The scalar
        uniform floor of the certified density is not a parameter
        distribution and is not sampled here.

        Returns
        -------
        samples : ndarray of shape (n_features (+ 1), n_samples)
            Parameters in original coordinates (with an intercept row when
            ``constant_feature='intercept'``).

        candidates : ndarray of shape (n_samples,)
            The candidate index of each draw.
        """
        pilot = self._check_pilot()
        q = self._weights(weights)
        rng = check_random_state(
            self.random_state if random_state is None else random_state
        )
        n_dim = pilot.ball_dim
        candidates = rng.choice(q.size, size=n_samples, p=q)
        g = rng.randn(n_dim, n_samples)
        g /= np.linalg.norm(g, axis=0, keepdims=True)
        radius = rng.uniform(size=n_samples) ** (1.0 / n_dim)
        u = g * radius
        theta = np.empty((n_dim, n_samples))
        # Center in the (augmented) original coordinates.
        if pilot.with_intercept:
            coef = pilot.whiten @ pilot.center[:-1]
            mu = np.concatenate([coef, pilot.center[-1:]])
        else:
            mu = pilot.whiten @ pilot.center
        for k in np.unique(candidates):
            sel = candidates == k
            theta[:, sel] = mu[:, None] + self._candidate_sqrt(int(k)) @ u[:, sel]
        if pilot.with_intercept:
            theta_f = theta[:-1]
            theta_i = theta[-1] + pilot.y_offset - pilot.x_offset @ theta_f
            theta = np.vstack([theta_f, theta_i])
        return theta, candidates
