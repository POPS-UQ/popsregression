"""
Estimator base class, input checks and the JAX configuration.

The estimators follow the familiar ``model.fit(X, y)`` / ``model.predict(X)``
usage and expose fitted attributes such as ``coef_``, but the package does not
depend on scikit-learn. ``get_params`` / ``set_params`` are provided because
they are cheap and make an estimator usable wherever an object is cloned from
its constructor parameters, which includes scikit-learn pipelines and
parameter searches when scikit-learn happens to be installed.

All numerical kernels run in JAX at double precision. Double precision is a
process-wide JAX setting, so importing this package enables it.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import copy
import inspect
import numbers

import jax
import numpy as np

# The barrier objective, the whitening and the low-noise evidence updates all
# lose their meaning in single precision.
jax.config.update("jax_enable_x64", True)


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit``."""


class BaseEstimator:
    """Constructor-parameter bookkeeping shared by the estimators."""

    @classmethod
    def _param_names(cls):
        parameters = inspect.signature(cls.__init__).parameters
        return [name for name in parameters if name != "self"]

    def get_params(self, deep=True):
        """Constructor parameters of this estimator, as a dict."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params):
        """Set constructor parameters; returns ``self``."""
        valid = self._param_names()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {valid}."
                )
            setattr(self, name, value)
        return self

    def __repr__(self):
        defaults = inspect.signature(type(self).__init__).parameters
        changed = []
        for name in self._param_names():
            value = getattr(self, name, None)
            default = defaults[name].default
            try:
                same = value is default or bool(np.all(value == default))
            except Exception:
                same = False
            if not same:
                changed.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(changed)})"

    def __sklearn_tags__(self):
        """Estimator tags, read by scikit-learn pipelines and searches.

        Only scikit-learn calls this method, so the import below never makes
        scikit-learn a requirement of this package.
        """
        from sklearn.utils import RegressorTags, Tags, TargetTags

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            regressor_tags=RegressorTags(),
        )

    def score(self, X, y):
        """Coefficient of determination of the mean prediction."""
        y = np.asarray(y, dtype=float).ravel()
        residual = y - np.asarray(self.predict(X))
        total = float(np.sum((y - y.mean()) ** 2))
        if total == 0.0:
            return 0.0
        return 1.0 - float(residual @ residual) / total


def check_is_fitted(estimator, attribute="coef_"):
    """Raise :class:`NotFittedError` unless ``attribute`` has been set."""
    if not hasattr(estimator, attribute):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' before using this estimator."
        )


def check_X(X, n_features=None):
    """A finite, two-dimensional float64 array."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must not be empty, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but the model was fitted with {n_features}."
        )
    return X


def check_X_y(X, y):
    """Validated ``(X, y)`` with one finite target per row."""
    X = check_X(X)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}.")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y have inconsistent lengths: {X.shape[0]} and {y.shape[0]}."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")
    return X, y


def check_sample_weight(sample_weight, n_samples):
    """Non-negative per-row weights; ``None`` means unit weights."""
    if sample_weight is None:
        return np.ones(n_samples)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full(n_samples, float(weights))
    if weights.shape != (n_samples,):
        raise ValueError(
            f"sample_weight must have shape ({n_samples},), got {weights.shape}."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and non-negative.")
    if not np.any(weights > 0):
        raise ValueError("sample_weight must have at least one positive entry.")
    return weights


def check_random_state(seed):
    """``None`` (NumPy global state), an int seed or a ``RandomState``."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a numpy RandomState.")


def check_scalar(name, value, low=None, high=None, low_open=False, high_open=False):
    """A real scalar inside the stated interval."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}.")
    below = low is not None and (value <= low if low_open else value < low)
    above = high is not None and (value >= high if high_open else value > high)
    if below or above or value != value:
        left = "(" if low_open else "["
        right = ")" if high_open else "]"
        low_text = "-inf" if low is None else low
        high_text = "inf" if high is None else high
        raise ValueError(
            f"{name} must lie in {left}{low_text}, {high_text}{right}, got {value!r}."
        )
    return value


def check_option(name, value, options):
    """A value from a fixed set of options."""
    if value not in options:
        raise ValueError(
            f"{name} must be one of {sorted(map(str, options))}, got {value!r}."
        )
    return value


def clone_unfitted(transformer):
    """An independent copy of a user-supplied transformer."""
    return copy.deepcopy(transformer)
