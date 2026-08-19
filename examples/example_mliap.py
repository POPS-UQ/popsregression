"""Posterior calibration of a linear Cu ACE interatomic potential.

The bundled ``ace_linear_uq_energies.npz`` holds the energy equations of a
linear 267-feature Cu ACE potential exported by ``mliap_train.py``: 700
training structures and 300 held-out test structures. The design is projected
onto its leading PCA modes so that small observation/parameter ratios are
reachable from 700 structures alone, and two regimes (N/P = 1.5 and N/P = 20)
are fitted with ``BayesianRidge``, the POPS hypercube, the POPS ellipse, and
the PAC-Bayes POPS ellipse.

The main figure is a probability-probability (P-P) plot: the posterior-sampled
CDF of the held-out energy error against the observed CDF, which is the parity
line for a perfectly calibrated posterior. Each panel is annotated with the
signed miscalibration area between the two (see
:func:`probability_probability_curve`). ``--error-output`` additionally writes
the two error densities that the P-P plot compares.

A trusted pickle with the same four keys can be supplied with ``--data``.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge

from popsregression import POPSRegression, POPSRegressionEllipse

SEED = 0
POSTERIOR_SAMPLE_COUNT = 1024
DATA_RATIOS = (1.5, 20.0)
PCA_VARIANCE = 0.95
PCA_MIN_RANK = 35
MAX_SIGMA = 4.0
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "ace_linear_uq_energies.npz"
DEFAULT_OUTPUT = HERE / "example_mliap.png"
DEFAULT_ERROR_OUTPUT = HERE / "example_mliap_errors.png"
DATA_KEYS = ("A_train_E", "y_train_E", "A_test_E", "y_test_E")
MODEL_TITLES = (
    "Bayesian Ridge",
    "POPS Hypercube",
    "POPS Ellipse",
    "POPS Ellipse + PAC",
)


def load_mliap_data(path):
    """Load and validate the ACE train/test energy equations."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            data = {key: archive[key] for key in DATA_KEYS}
    elif path.suffix in (".pkl", ".pickle"):
        # Pickle support is for the artifact produced by the accompanying
        # workflow. As usual, only load trusted pickle files.
        with path.open("rb") as stream:
            archive = pickle.load(stream)
        data = {key: archive[key] for key in DATA_KEYS}
    else:
        raise ValueError("--data must be an .npz, .pkl, or .pickle file")

    X_train = np.asarray(data["A_train_E"], dtype=np.float64)
    y_train = np.asarray(data["y_train_E"], dtype=np.float64).reshape(-1)
    X_test = np.asarray(data["A_test_E"], dtype=np.float64)
    y_test = np.asarray(data["y_test_E"], dtype=np.float64).reshape(-1)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("ACE design matrices must be two-dimensional")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("train and test design matrices have different widths")
    if X_train.shape[0] != y_train.size or X_test.shape[0] != y_test.size:
        raise ValueError("each design-matrix row must have one target")
    arrays = (X_train, y_train, X_test, y_test)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("ACE data contain non-finite values")
    return X_train, y_train, X_test, y_test


def pca_basis(X):
    """Leading PCA modes of the training design.

    Enough modes are kept to explain ``PCA_VARIANCE`` of the variance, but
    never fewer than ``PCA_MIN_RANK``. Reducing the rank is what makes small
    N/P ratios reachable from the 700 bundled training structures.
    """
    centered = X - X.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    rank = int(np.searchsorted(cumulative, PCA_VARIANCE, side="left") + 1)
    rank = min(max(rank, PCA_MIN_RANK), X.shape[1])
    return eigenvectors[:, order[:rank]]


def make_regimes(X_train, y_train, basis, ratios=DATA_RATIOS):
    """Draw one training subset per requested observation/parameter ratio."""
    n_train, rank = len(y_train), basis.shape[1]
    projected = X_train @ basis
    max_feasible_ratio = n_train / rank  # Cap to the available structures
    rng = np.random.default_rng(SEED)

    regimes = []
    for ratio in ratios:
        ratio = min(ratio, max_feasible_ratio)
        n_samples = max(2, int(round(ratio * rank)))
        indices = rng.choice(n_train, n_samples, replace=False)
        regimes.append(
            {
                "ratio": float(ratio),
                "X": projected[indices],
                "y": y_train[indices],
            }
        )
    return tuple(regimes)


def fit_models(X_train, y_train):
    """Fit the same four estimators as the polynomial and Burgers examples."""
    # Use a fixed-size hypercube sample in every data regime. The current
    # POPSRegression implementation draws from NumPy's global RNG.
    np.random.seed(SEED)
    resample_density = POSTERIOR_SAMPLE_COUNT / len(y_train)

    bayesian_ridge = BayesianRidge(fit_intercept=False)
    bayesian_ridge.fit(X_train, y_train)

    pops_hypercube = POPSRegression(
        fit_intercept=False,
        minimum_relative_error=0.0,
        posterior="hypercube",
        resample_density=resample_density,
    )
    pops_hypercube.fit(X_train, y_train)

    pops_ellipse = POPSRegressionEllipse(random_state=SEED)
    pops_ellipse.fit(X_train, y_train)

    # PAC-Bayes variant: phase-1 hyperprior centre with a tight scale, and a
    # short rho schedule for faster convergence.
    pops_ellipse_pac = POPSRegressionEllipse(
        random_state=SEED,
        pac_bayes=True,
        hyperprior_center="phase1",
        hyperprior_scale=1.0,
        rho_schedule=[1.0, 0.1, 0.01],
    )
    pops_ellipse_pac.fit(X_train, y_train)

    return (bayesian_ridge, pops_hypercube, pops_ellipse, pops_ellipse_pac)


def sample_bayesian_errors(model, X, rng, n_samples):
    """Draw prediction errors from the Bayesian coefficient posterior."""
    eigenvalues, eigenvectors = np.linalg.eigh(model.sigma_)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    coefficient_errors = factor @ rng.standard_normal((factor.shape[1], n_samples))
    return X @ coefficient_errors


def projected_ball_samples(rng, n_rows, n_samples, dimension):
    """Sample one-dimensional marginals of a uniform dimension-D ball."""
    beta_shape = 0.5 * (dimension + 1.0)
    return 2.0 * rng.beta(beta_shape, beta_shape, (n_rows, n_samples)) - 1.0


def sample_ellipse_errors(model, X, rng, n_samples):
    """Draw exact marginal errors from a bare or PAC-Bayes ellipse.

    For the bare ellipse this is the exact projected-ball pushforward. For the
    PAC model, the diagonal Laplace hyperposterior is sampled first and then a
    projected-ball error is drawn conditionally. Independent hyperposterior
    draws per test row preserve the marginal densities needed by these plots.
    """
    Xc, Z = model._whitened_design(np.asarray(X, dtype=np.float64))
    projected_factor = Z @ model.U_
    fitted_width = model._squared_widths(Xc, Z) + model.delta**2
    baseline_width = np.maximum(
        fitted_width - np.sum(projected_factor**2, axis=1),
        0.0,
    )

    if not model._pac_bayes_fitted:
        ball = projected_ball_samples(rng, len(X), n_samples, model._ball_dim)
        return np.sqrt(fitted_width)[:, None] * ball

    Z2 = Z * Z
    projected_variance = np.maximum(Z2 @ model._sigma_U, 0.0)
    center_variance = np.maximum(Z2 @ model._sigma_c, 0.0)
    errors = np.empty((len(X), n_samples))
    chunk_size = 32
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        mean_projection = projected_factor[start:stop, None, :]
        projection_scale = np.sqrt(projected_variance[start:stop, None, :])
        sampled_projection = mean_projection + projection_scale * rng.standard_normal(
            (stop - start, n_samples, model.rank_)
        )
        sampled_width = baseline_width[start:stop, None] + np.sum(
            sampled_projection**2,
            axis=2,
        )
        ball = projected_ball_samples(
            rng,
            stop - start,
            n_samples,
            model._ball_dim,
        )
        center_error = np.sqrt(center_variance[start:stop, None]) * (
            rng.standard_normal((stop - start, n_samples))
        )
        errors[start:stop] = center_error + np.sqrt(sampled_width) * ball
    return errors


def sample_posterior_errors(model, X, rng):
    """Return posterior prediction errors about the model's point prediction."""
    if isinstance(model, BayesianRidge) and not isinstance(model, POPSRegression):
        return sample_bayesian_errors(model, X, rng, POSTERIOR_SAMPLE_COUNT)
    if isinstance(model, POPSRegression):
        # POPSRegression.posterior_samples_ contains coefficient perturbations,
        # not absolute coefficient vectors. Therefore no mean prediction is
        # subtracted here.
        return X @ model.posterior_samples_
    return sample_ellipse_errors(model, X, rng, POSTERIOR_SAMPLE_COUNT)


def probability_probability_curve(observed_errors, posterior_errors):
    """Empirical P-P curve of |error| and its signed miscalibration area.

    Plotting one empirical CDF against another is a probability-probability
    (P-P) plot; a perfectly calibrated posterior lies on the parity line. The
    signed area enclosed between the curve and that line,

        A = int_0^1 F_post(F_obs^-1(u)) du - 1/2
          = P(|e_post| < |e_obs|) - 1/2,

    is the *signed miscalibration area*. It is the Mann-Whitney statistic
    shifted to zero, so 2A is the Gini coefficient (equivalently Somers' D, or
    twice the ROC excess area AUC - 1/2) of the two error samples. It lies in
    [-1/2, +1/2], is invariant under any common rescaling of the errors, and is
    zero exactly when the two distributions agree. Positive values mean the
    posterior errors are stochastically smaller than the observed ones, i.e.
    the posterior is too narrow (over-confident); negative values mean it is
    too wide.

    Returns the observed CDF, the posterior CDF, and the signed area. The area
    is integrated over the returned curve, so it is exactly the area drawn.
    """
    observed = np.sort(np.abs(np.asarray(observed_errors, dtype=np.float64).ravel()))
    posterior = np.sort(np.abs(np.asarray(posterior_errors, dtype=np.float64).ravel()))
    if observed.size == 0 or posterior.size == 0:
        raise ValueError("both error samples must be non-empty")
    thresholds = np.unique(np.concatenate([[0.0], observed, posterior]))
    observed_cdf = np.searchsorted(observed, thresholds, side="right") / observed.size
    posterior_cdf = (
        np.searchsorted(posterior, thresholds, side="right") / posterior.size
    )
    area = float(np.trapezoid(posterior_cdf, observed_cdf) - 0.5)
    return observed_cdf, posterior_cdf, area


def posterior_error_records(models, X_test, y_test):
    """Collect observed and sampled posterior energy errors for each model."""
    records = []
    rng = np.random.default_rng(SEED)
    for model in models:
        test_errors = y_test - model.predict(X_test)
        pred_errors = sample_posterior_errors(model, X_test, rng)
        posterior_rms = np.sqrt(np.mean(pred_errors**2, axis=1))
        if np.any(posterior_rms <= 0.0):
            raise RuntimeError("encountered a zero energy posterior width")
        posterior_95 = np.sqrt(np.percentile(pred_errors**2, 95, axis=1))
        _, _, area = probability_probability_curve(test_errors, pred_errors)
        records.append(
            {
                "test_errors": test_errors,
                "pred_errors": pred_errors,
                "posterior_rms": posterior_rms,
                "miscalibration_area": area,
                "actual_coverage": float(np.mean(np.abs(test_errors) <= posterior_95)),
                "energy_rmse": float(np.sqrt(np.mean(test_errors**2))),
            }
        )
    return records


ANNOTATION_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="0.65", alpha=0.90)


def area_label(record):
    """Annotation text for a panel's signed miscalibration area."""
    return f"misc. area = {record['miscalibration_area']:+.3f}"


def make_panel_grid(n_rows):
    """Shared 4-column panel grid with model titles on the top row."""
    fig, axes = plt.subplots(n_rows, 4, figsize=(8, 3), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for column, title in enumerate(MODEL_TITLES):
        axes[0, column].set_title(title, fontsize=10)
    return fig, axes


def common_error_scale(all_records):
    """Use the largest-N Bayesian-Ridge RMS as a common density x-axis scale."""
    scale = np.asarray(all_records[-1][0]["posterior_rms"], dtype=np.float64)
    return float(np.sqrt(np.mean(scale**2)))


def plot_error_densities(regimes, all_records, output):
    """Overlay the observed and posterior-sampled error densities."""
    scale = common_error_scale(all_records)
    max_ratio = max(
        np.max(np.abs(record["test_errors"]) / scale)
        for records in all_records
        for record in records
    )
    x_max = min(10.0, max(MAX_SIGMA, 1.05 * max_ratio))
    bins = np.linspace(-x_max, x_max, 31)

    fig, axes = make_panel_grid(len(regimes))
    for row, (regime, records) in enumerate(zip(regimes, all_records)):
        for column, record in enumerate(records):
            ax = axes[row, column]
            ax.hist(
                (record["pred_errors"] / scale).ravel(),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color="C1",
                label="Posterior",
            )
            ax.hist(
                record["test_errors"] / scale,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color="0.15",
                label="Observed",
            )
            ax.text(
                0.96,
                0.92,
                area_label(record),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.2,
                bbox=ANNOTATION_BBOX,
            )
            ax.set_xlim(-x_max, x_max)
            ax.set_ylim(bottom=0.0, top=0.5)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            if row == len(regimes) - 1:
                ax.set_xlabel(r"$(E-E_{\rm DFT})/\sigma_{\rm RMS}$", fontsize=8)
        axes[row, 0].set_ylabel(f"N/P = {regime['ratio']:.1f}", fontsize=8)

    axes[0, -1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout(pad=0.25, w_pad=0.2, h_pad=0.25)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pp(regimes, all_records, output):
    """P-P plot of posterior-predicted against observed |energy error| CDFs.

    The curve is scale-free, so no common RMS normalisation is needed here.
    """
    fig, axes = make_panel_grid(len(regimes))
    for row, (regime, records) in enumerate(zip(regimes, all_records)):
        for column, record in enumerate(records):
            ax = axes[row, column]
            observed_cdf, posterior_cdf, _ = probability_probability_curve(
                record["test_errors"],
                record["pred_errors"],
            )
            ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.5, label="Calibrated")
            ax.plot(
                observed_cdf,
                posterior_cdf,
                "C1",
                linewidth=1.5,
                label="Posterior vs observed",
            )
            ax.fill_between(
                observed_cdf,
                observed_cdf,
                posterior_cdf,
                color="C1",
                alpha=0.15,
                linewidth=0.0,
            )
            ax.text(
                0.96,
                0.08,
                area_label(record),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.2,
                bbox=ANNOTATION_BBOX,
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            # Prune the 0/1 x-ticks so neighbouring panels do not collide.
            ax.set_xticks([0.25, 0.5, 0.75])
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(labelsize=7)
            if row == len(regimes) - 1:
                ax.set_xlabel("Observed CDF", fontsize=8)
        axes[row, 0].set_ylabel(
            f"N/P = {regime['ratio']:.1f}\nPosterior CDF",
            fontsize=8,
        )

    axes[0, -1].legend(fontsize=6, loc="upper left")
    fig.tight_layout(pad=0.25, w_pad=0.2, h_pad=0.25)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(data=DEFAULT_DATA, output=DEFAULT_OUTPUT, error_output=None):
    X_train, y_train, X_test, y_test = load_mliap_data(data)
    basis = pca_basis(X_train)
    regimes = make_regimes(X_train, y_train, basis)
    X_test_projected = X_test @ basis
    print(
        f"ACE energies: train={len(y_train)}, test={len(y_test)}, "
        f"P={X_train.shape[1]} reduced to rank {basis.shape[1]}"
    )

    all_records = [
        posterior_error_records(
            fit_models(regime["X"], regime["y"]),
            X_test_projected,
            y_test,
        )
        for regime in regimes
    ]

    plot_pp(regimes, all_records, output)
    print(f"Saved {Path(output).resolve()}")
    if error_output:
        plot_error_densities(regimes, all_records, error_output)
        print(f"Saved {Path(error_output).resolve()}")

    for regime, records in zip(regimes, all_records):
        print(f"N={len(regime['y'])}, N/P={regime['ratio']:.3f}")
        for title, record in zip(MODEL_TITLES, records):
            print(
                f"  {title:20s} signed miscalibration area="
                f"{record['miscalibration_area']:+.4f}  "
                "95% posterior interval covers "
                f"{record['actual_coverage'] * 100:.1f}% of tests  "
                f"RMSE E={record['energy_rmse']:.4g} eV/atom"
            )
    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="ACE energy equations as NPZ or trusted pickle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="P-P (CDF vs CDF) figure path",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        nargs="?",
        const=DEFAULT_ERROR_OUTPUT,
        default=None,
        help="also write the error-density figure (optional path)",
    )
    args = parser.parse_args()
    run(data=args.data, output=args.output, error_output=args.error_output)
