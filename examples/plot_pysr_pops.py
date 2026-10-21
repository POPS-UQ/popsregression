"""
=====================================================
POPS uncertainty for PySR symbolic regression models
=====================================================

`PySR <https://github.com/MilesCranmer/PySR>`_ returns point-estimate
formulas: there is no uncertainty on the fitted constants or predictions.
A formula from symbolic regression is misspecified by construction (it is
the best expression found within a complexity budget), so in the low-noise
regime its error is dominated by misspecification, which is exactly what
:class:`~popsregression.POPSRegression` estimates.

POPS is linear in its parameters, so we linearise the formula
:math:`f(x;\\theta)` about its fitted constants :math:`\\theta^*`,

.. math::

    f(x;\\theta^* + \\delta\\theta) \\approx f(x;\\theta^*)
    + J(x)\\,\\delta\\theta, \\qquad J(x) = \\partial_\\theta f(x;\\theta^*),

and fit POPS to the residuals :math:`y - f(x;\\theta^*)` with design matrix
:math:`J`. The POPS posterior over :math:`\\delta\\theta` is then pushed
back through the full nonlinear formula. For formulas that are linear in
their constants this is exact.

Running PySR needs a Julia install; if ``pysr`` is not importable this
example falls back to a hand-written expression of the kind PySR returns.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>,
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Low-noise training data
# -----------------------

import numpy as np

rng = np.random.RandomState(42)


def target_function(x):
    return (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0


x_train = np.sort(rng.uniform(-10, 10, 100))
y_train = target_function(x_train)
x_test = np.linspace(-11, 11, 200)
y_test = target_function(x_test)

# %%
# Find a formula with PySR
# ------------------------
#
# We cap the complexity so that the formula cannot reproduce the target
# exactly. ``model.sympy()`` returns the selected equation as a sympy
# expression in the feature names (here ``x0``).

import sympy

try:
    from pysr import PySRRegressor

    sr = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin"],
        maxsize=12,
        random_state=0,
        deterministic=True,
        parallelism="serial",
    )
    sr.fit(x_train.reshape(-1, 1), y_train)
    expr = sr.sympy()
except ImportError:
    # A typical PySR result at this complexity: it misses the x**4 term.
    expr = sympy.sympify("9.8*x0*sin(0.99*x0) + 0.12*x0**3")

print("PySR formula:", expr)

# %%
# Linearise the formula in its constants
# --------------------------------------
#
# Every floating point number in the expression is a fitted constant. We
# promote them to symbols, refit them by least squares (PySR's constants are
# already optimised, but the linearisation assumes a stationary point) and
# build the Jacobian with respect to the constants.

from scipy.optimize import least_squares


def linearise(expr, n_features):
    """Return ``f(X, theta)``, ``J(X, theta)`` and initial constants."""
    x_syms = sympy.symbols(f"x0:{n_features}")
    floats = sorted(expr.atoms(sympy.Float), key=str)
    theta_syms = sympy.symbols(f"theta0:{len(floats)}")
    param_expr = expr.xreplace(dict(zip(floats, theta_syms)))
    grad = [sympy.diff(param_expr, t) for t in theta_syms]

    f_ = sympy.lambdify((x_syms, theta_syms), param_expr, "numpy")
    g_ = sympy.lambdify((x_syms, theta_syms), grad, "numpy")

    def f(X, theta):
        return np.broadcast_to(f_(X.T, theta), X.shape[:1])

    def jac(X, theta):
        return np.column_stack(
            [np.broadcast_to(g, X.shape[:1]) for g in g_(X.T, theta)]
        )

    return f, jac, np.array(floats, dtype=float)


X_train = x_train.reshape(-1, 1)
X_test = x_test.reshape(-1, 1)

f, jac, theta0 = linearise(expr, n_features=1)
theta_star = least_squares(
    lambda t: f(X_train, t) - y_train, theta0, jac=lambda t: jac(X_train, t)
).x
print("Fitted constants:", theta_star)

# %%
# Fit POPS on the Jacobian
# ------------------------

from popsregression import POPSRegression

pops = POPSRegression(resampling_method="sobol", resample_density=10.0)
pops.fit(jac(X_train, theta_star), y_train - f(X_train, theta_star))

# Linearised prediction and uncertainty
J_test = jac(X_test, theta_star)
dy_mean, y_std = pops.predict(J_test, return_std=True)
y_pred = f(X_test, theta_star) + dy_mean

# Nonlinear propagation: evaluate the formula at each posterior sample
theta_samples = theta_star + pops.coef_ + pops.posterior_samples_.T
y_samples = np.array([f(X_test, t) for t in theta_samples])
y_min, y_max = y_samples.min(axis=0), y_samples.max(axis=0)

# %%
# Plot
# ----

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
ax.fill_between(x_test, y_min, y_max, alpha=0.2, facecolor="0.5", label="POPS max/min")
ax.fill_between(
    x_test,
    y_pred - 2 * y_std,
    y_pred + 2 * y_std,
    alpha=0.5,
    facecolor="C1",
    label=r"POPS $\pm 2\sigma$",
)
ax.plot(x_test, y_pred, "C1-", lw=3, label="PySR formula")
ax.plot(x_train, y_train, "b.", ms=4, label="Train")
ax.plot(x_test, y_test, "k-", lw=1, label="Truth")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"$y = {sympy.latex(expr)}$", fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# The PySR formula has no ``x**4`` term, so it drifts from the truth at
# large ``|x|``. The POPS band widens there: the constants that would fit
# those points exactly differ most from the global fit.
