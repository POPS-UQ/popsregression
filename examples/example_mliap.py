"""Posterior calibration of a linear Cu ACE interatomic potential.

The bundled ``ace_linear_uq_energies.npz`` holds the energy equations of a
linear 267-feature Cu ACE potential exported by ``mliap_train.py``: 700
training structures and 300 held-out test structures. The design is projected
onto its leading 35 PCA modes plus a constant column (P = 36) so that small
observation/parameter ratios are reachable from 700 structures alone, and two
regimes (N/P = 1.5 and N/P = 20) are fitted with ``BayesianRidge``, the POPS
hypercube, POPS Ellipse, POPS Ellipse+EB and POPS Ellipse+PAC (paper Fig.
``fig:ace``).

By default (``--basis subset``) the PCA basis is learned from training
descriptors only: once from the whole training subset for the first four
methods, and separately on each pilot split inside POPS Ellipse+PAC, so its
certification structures never influence the features. ``--basis pool``
reproduces the workshop construction, whose basis used the unlabelled
descriptors of all 700 training structures (not valid for the PAC bound).
Every posterior is sampled for parameter uncertainty only (Bayesian ridge
``sigma_``, never its noise precision).

The main figure is a probability-probability (P-P) plot of pooled absolute
errors: the predicted distribution function of ``|E - E_mean|`` over all test
structures against the observed one; a perfectly matched spread lies on the
parity line. This is pooled error-distribution agreement, not calibration of
each structure's interval (see :mod:`comparisons.calibration`). Each panel is
annotated with the unsigned area ``A_abs`` and signed area ``A_s`` between the
curve and the parity line (see :func:`probability_probability_curve`).
``--error-output`` additionally writes the two error densities that the P-P
plot compares.

A trusted pickle with the same four keys can be supplied with ``--data``.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from comparisons import calibration, harness
from comparisons.labels import display
from sklearn.linear_model import BayesianRidge

from popsregression import POPSEllipseRegression, POPSRegression

SEED = 0
POSTERIOR_SAMPLE_COUNT = 1024
DATA_RATIOS = (1.5, 20.0)
MAX_SIGMA = 4.0
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "ace_linear_uq_energies.npz"
DEFAULT_OUTPUT = HERE / "example_mliap.png"
DEFAULT_ERROR_OUTPUT = HERE / "example_mliap_errors.png"
DATA_KEYS = ("A_train_E", "y_train_E", "A_test_E", "y_test_E")
MODEL_TITLES = tuple(
    display(m)
    for m in (
        "Bayesian ridge",
        "POPS hypercube",
        "POPS ellipse",
        "Ellipse+EB",
        "Ellipse+PAC",
    )
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


def make_regimes(ratios=DATA_RATIOS, basis="subset"):
    """One training subset per observation/parameter ratio (35 PCA modes)."""
    regimes = []
    for ratio in ratios:
        problem = harness.ace_problem(ratio, SEED, basis=basis)
        regimes.append(
            {
                "ratio": problem.y_train.size / problem.n_params,
                "problem": problem,
                "y": problem.y_train,
                "y_test": problem.y_test,
            }
        )
    return tuple(regimes)


def fit_models(problem):
    """Fit the same five estimators as the polynomial and Burgers examples.

    Returns ``(model, X_eval)`` pairs: Bayesian ridge and the hypercube use
    the PCA features fitted on the whole training subset; the ellipse family
    takes the raw descriptors and its own ``preprocessor`` (for PAC fitted on
    each pilot split only).
    """
    X_train, X_test, _ = problem.design()
    y_train = problem.y_train
    resample_density = POSTERIOR_SAMPLE_COUNT / len(y_train)

    bayesian_ridge = BayesianRidge(fit_intercept=False)
    bayesian_ridge.fit(X_train, y_train)

    pops_hypercube = POPSRegression(
        fit_intercept=False,
        minimum_relative_error=0.0,
        posterior="hypercube",
        resample_density=resample_density,
        random_state=SEED,
    )
    pops_hypercube.fit(X_train, y_train)

    raw, raw_test, pre = problem.X_train, problem.X_test, problem.preprocessor
    ellipse = POPSEllipseRegression(preprocessor=pre, random_state=SEED).fit(
        raw, y_train
    )
    ellipse_eb = POPSEllipseRegression(
        regularization="empirical-bayes", preprocessor=pre, random_state=SEED
    ).fit(raw, y_train)
    ellipse_pac = POPSEllipseRegression(
        regularization="PAC",
        y_bounds=harness.ACE_Y_BOUNDS,
        preprocessor=pre,
        random_state=SEED,
    ).fit(raw, y_train)
    return (
        (bayesian_ridge, X_test),
        (pops_hypercube, X_test),
        (ellipse, raw_test),
        (ellipse_eb, raw_test),
        (ellipse_pac, raw_test),
    )


def sample_bayesian_errors(model, X, rng, n_samples):
    """Draw prediction errors from the Bayesian coefficient posterior."""
    eigenvalues, eigenvectors = np.linalg.eigh(model.sigma_)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    coefficient_errors = factor @ rng.standard_normal((factor.shape[1], n_samples))
    return X @ coefficient_errors


def sample_ellipse_errors(model, X, rng, n_samples):
    """Errors of joint draws of the (hierarchical) ellipsoid predictive.

    Each column is one parameter draw ``(theta, a)`` of the returned
    predictive (offset coordinate included), so the marginal at each input is
    exactly the scored predictive; errors are measured from the mean.
    """
    draws = model.sample_predictions(
        X, n_samples, random_state=int(rng.integers(2**31 - 1))
    )
    return draws - model.predict(X)[:, None]


def sample_posterior_errors(model, X, rng):
    """Return posterior prediction errors about the model's point prediction."""
    if isinstance(model, POPSEllipseRegression):
        return sample_ellipse_errors(model, X, rng, POSTERIOR_SAMPLE_COUNT)
    if isinstance(model, POPSRegression):
        # POPSRegression.posterior_samples_ contains coefficient perturbations,
        # not absolute coefficient vectors. Therefore no mean prediction is
        # subtracted here.
        return X @ model.posterior_samples_
    return sample_bayesian_errors(model, X, rng, POSTERIOR_SAMPLE_COUNT)


def probability_probability_curve(observed_errors, posterior_errors):
    """Pooled P-P curve of |error| and its signed and unsigned areas.

    Plotting the predicted distribution function of the pooled absolute
    errors against the observed one is a probability-probability (P-P) plot,
    ``C(u) = F_post(F_obs^{-1}(u))``; matching error distributions lie on the
    parity line. The two areas are

        A_s   = int_0^1 (C(u) - u) du = P(|e_post| < |e_obs|) - 1/2,
        A_abs = int_0^1 |C(u) - u| du.

    ``A_s`` is the Mann-Whitney statistic shifted to zero (so ``2 A_s`` is
    the Gini coefficient, or Somers' D, of the two samples); it lies in
    ``[-1/2, 1/2]`` and is invariant under a common rescaling of the errors.
    ``A_s = 0`` does NOT mean the two distributions agree (positive and
    negative parts of ``C(u) - u`` can cancel), and ``A_s > 0`` alone does NOT
    mean the predicted errors are stochastically smaller (that requires
    ``C(u) >= u`` for every ``u``); ``A_s > 0`` indicates net
    over-confidence and ``A_s < 0`` net under-confidence. ``A_abs`` is zero
    exactly when the two pooled distributions agree, and is the primary
    ranking (smaller is better). Pooled agreement is not per-input
    calibration: it can hold while individual intervals are wrong.

    Returns the observed CDF ``u``, the curve ``C(u)`` (a step function,
    constant on each ``(u[k-1], u[k]]``), ``A_s`` and ``A_abs``, integrated
    exactly over the returned curve.
    """
    posterior = np.asarray(posterior_errors, dtype=np.float64)
    observed = np.asarray(observed_errors, dtype=np.float64).ravel()
    if observed.size == 0 or posterior.size == 0:
        raise ValueError("both error samples must be non-empty")
    return calibration.error_pp_curve(observed, posterior)


def posterior_error_records(models, y_test):
    """Collect observed and sampled posterior energy errors for each model."""
    records = []
    rng = np.random.default_rng(SEED)
    for model, X_test in models:
        test_errors = y_test - model.predict(X_test)
        pred_errors = sample_posterior_errors(model, X_test, rng)
        posterior_rms = np.sqrt(np.mean(pred_errors**2, axis=1))
        if np.any(posterior_rms <= 0.0):
            raise RuntimeError("encountered a zero energy posterior width")
        posterior_95 = np.sqrt(np.percentile(pred_errors**2, 95, axis=1))
        _, _, area, area_abs = probability_probability_curve(test_errors, pred_errors)
        records.append(
            {
                "test_errors": test_errors,
                "pred_errors": pred_errors,
                "posterior_rms": posterior_rms,
                "miscalibration_area": area,
                "miscalibration_area_abs": area_abs,
                "actual_coverage": float(np.mean(np.abs(test_errors) <= posterior_95)),
                "energy_rmse": float(np.sqrt(np.mean(test_errors**2))),
            }
        )
    return records


ANNOTATION_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="0.65", alpha=0.90)


def area_label(record):
    """Annotation text for a panel's unsigned and signed areas."""
    return (
        f"$A_{{abs}}$ = {record['miscalibration_area_abs']:.3f}, "
        f"$A_s$ = {record['miscalibration_area']:+.3f}"
    )


def make_panel_grid(n_rows):
    """Shared panel grid with model titles on the top row."""
    fig, axes = plt.subplots(
        n_rows, len(MODEL_TITLES), figsize=(10, 3.4), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for column, title in enumerate(MODEL_TITLES):
        axes[0, column].set_title(title, fontsize=9.5, pad=12)
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
            observed_cdf, posterior_cdf, _, _ = probability_probability_curve(
                record["test_errors"],
                record["pred_errors"],
            )
            # The curve is constant on each (u[k-1], u[k]]: evaluate it on a
            # fine grid so the shaded region is exactly the integrated one.
            grid = np.linspace(0.0, 1.0, 2001)
            curve = posterior_cdf[
                np.clip(np.searchsorted(observed_cdf, grid), 0, grid.size)
            ]
            ax.plot(
                [0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.5, label="Matched spread"
            )
            ax.plot(grid, curve, "C1", linewidth=1.5, label="Predicted vs observed")
            ax.fill_between(grid, grid, curve, color="C1", alpha=0.15, linewidth=0.0)
            ax.text(
                0.5,
                1.015,
                area_label(record),
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            # Prune the 0/1 x-ticks so neighbouring panels do not collide.
            ax.set_xticks([0.25, 0.5, 0.75])
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(labelsize=7)
            if row == len(regimes) - 1:
                ax.set_xlabel("Observed CDF of |error|", fontsize=8)
        axes[row, 0].set_ylabel(
            f"N/P = {regime['ratio']:.1f}\nPredicted CDF of |error|",
            fontsize=8,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(pad=0.25, w_pad=0.3, h_pad=0.9, rect=(0, 0.06, 1, 1))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(data=DEFAULT_DATA, output=DEFAULT_OUTPUT, error_output=None, basis="subset"):
    X_train, y_train, X_test, y_test = load_mliap_data(data)
    regimes = make_regimes(basis=basis)
    print(
        f"ACE energies: train={len(y_train)}, test={len(y_test)}, "
        f"P={X_train.shape[1]} reduced to rank {harness.ACE_RANK} "
        f"(PCA basis: {basis})"
    )

    all_records = [
        posterior_error_records(fit_models(regime["problem"]), regime["y_test"])
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
                f"  {title:20s} A_abs={record['miscalibration_area_abs']:.4f} "
                f"A_s={record['miscalibration_area']:+.4f}  "
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
    parser.add_argument(
        "--basis",
        choices=("subset", "pool"),
        default="subset",
        help="PCA basis from the training subset (default) or all 700 structures",
    )
    args = parser.parse_args()
    run(
        data=args.data,
        output=args.output,
        error_output=args.error_output,
        basis=args.basis,
    )
