"""Polynomial workshop figure in the Burgers-style row/column layout.

Rows are training-set sizes; columns are BayesianRidge, POPS Hypercube,
POPS Ellipse, and POPS Ellipse + PAC.  The visual convention matches the
Burgers example: an outer high-confidence/support band and a darker inner
band, with truth and training data overlaid.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from popsregression import POPSRegression

SEED = 1042

# Match the Burgers/POD workshop figure styling.
OUTER_COLOR = "#8FA4BF"
OUTER_ALPHA = 0.42
INNER_COLOR = "C1"
INNER_ALPHA = 0.45
COVERAGE_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="0.65", alpha=0.90)


def target_function(x):
    return (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0


def generate_data(rng, n_samples):
    x_train = np.sort(
        np.append(rng.uniform(-1, 1, n_samples), np.linspace(-1, 1, 2)) * 10
    )
    x_dense = np.linspace(-10, 10, 201)
    y_dense = target_function(x_dense)

    poly = PolynomialFeatures(degree=4, include_bias=True)
    X_train = poly.fit_transform(x_train.reshape(-1, 1))
    X_dense = poly.transform(x_dense.reshape(-1, 1))
    y_train = target_function(x_train)
    return X_train, x_train, y_train, X_dense, x_dense, y_dense


def epistemic_std(model, X):
    return np.sqrt(np.sum((X @ model.sigma_) * X, axis=1))


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def draw_common(
    ax,
    x,
    truth,
    x_train,
    y_train,
    mean,
    truth_label="_nolegend_",
    mean_label="_nolegend_",
):
    ax.plot(x, truth, "k-", lw=1.5, label=truth_label)
    ax.plot(x, mean, "C1-", lw=2, label=mean_label)
    ax.plot(x_train, y_train, "b.", ms=4, label="_nolegend_")


def main():
    # POPSRegression resamples from NumPy's global RNG, so seed it as well as
    # the local generator to keep the committed figure reproducible.
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    train_sizes = (10, 100)
    titles = ["Bayesian Ridge", "POPS Hypercube", "POPS Ellipse", "POPS Ellipse + PAC"]

    fig, axes = plt.subplots(
        len(train_sizes), 4, figsize=(8, 3), sharex=True, sharey=True
    )

    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10)

    for row, n_samples in enumerate(train_sizes):
        X_train, x_train, y_train, X_dense, x_dense, y_dense = generate_data(
            rng, n_samples
        )

        bay = BayesianRidge(fit_intercept=False).fit(X_train, y_train)
        hyc = POPSRegression(minimum_relative_error=0.0, posterior="hypercube").fit(
            X_train, y_train
        )
        ell = POPSRegression(posterior="ellipsoid", random_state=0).fit(
            X_train, y_train
        )
        pac = POPSRegression(posterior="ellipsoid", pac_bayes=True, random_state=0).fit(
            X_train, y_train
        )

        # BayesianRidge: +/-4 sigma outer, +/-2 sigma inner.
        mean = bay.predict(X_dense)
        std = epistemic_std(bay, X_dense)
        b_lo = mean - 4 * std
        b_hi = mean + 4 * std
        b_cov = coverage(y_dense, b_lo, b_hi)

        ax = axes[row, 0]
        ax.fill_between(
            x_dense,
            b_lo,
            b_hi,
            alpha=OUTER_ALPHA,
            facecolor=OUTER_COLOR,
            label=r"max/min ($\pm4\sigma$)",
        )
        ax.fill_between(
            x_dense,
            mean - 2 * std,
            mean + 2 * std,
            alpha=INNER_ALPHA,
            facecolor=INNER_COLOR,
            label=r"$95.45\%$ ($\pm2\sigma$)",
        )
        draw_common(ax, x_dense, y_dense, x_train, y_train, mean, mean_label="mean")

        # POPS variants: full max/min outer band and +/-2 predictive std inner.
        pop_cov = {}
        for col, key, model in (
            (1, "hyper", hyc),
            (2, "ellipse", ell),
            (3, "pac", pac),
        ):
            mean, std, hi, lo = model.predict(
                X_dense, return_std=True, return_bounds=True
            )
            pop_cov[key] = coverage(y_dense, lo, hi)
            ax = axes[row, col]
            ax.fill_between(
                x_dense,
                lo,
                hi,
                alpha=OUTER_ALPHA,
                facecolor=OUTER_COLOR,
                label="max/min",
            )
            ax.fill_between(
                x_dense,
                mean - 2 * std,
                mean + 2 * std,
                alpha=INNER_ALPHA,
                facecolor=INNER_COLOR,
                label=r"$95.45\%$",
            )
            draw_common(
                ax, x_dense, y_dense, x_train, y_train, mean, truth_label="Truth"
            )

        # Coverage of the outer interval over the dense evaluation grid.
        coverage_text = [
            rf"$4\sigma$ cov. = {b_cov:.3f}",
            f"cov. = {pop_cov['hyper']:.3f}",
            f"cov. = {pop_cov['ellipse']:.3f}",
            f"cov. = {pop_cov['pac']:.3f}",
        ]
        for col, text in enumerate(coverage_text):
            axes[row, col].text(
                0.5,
                0.95,
                text,
                transform=axes[row, col].transAxes,
                ha="center",
                va="top",
                fontsize=6,
                bbox=COVERAGE_BBOX,
                zorder=10,
            )

        axes[row, 0].set_ylabel(f"N = {n_samples}", fontsize=9)

    for ax in axes.flat:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-150, 150)
        ax.tick_params(labelsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("x", fontsize=9)

    # Exactly two legends for readability. BayesianRidge explains the
    # Gaussian bands; a single POPS panel explains truth and POPS bands.
    bay_ax = axes[0, 0]
    handles, labels = bay_ax.get_legend_handles_labels()
    order = [
        labels.index("mean"),
        labels.index(r"$95.45\%$ ($\pm2\sigma$)"),
        labels.index(r"max/min ($\pm4\sigma$)"),
    ]
    bay_ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize=6,
        loc="lower center",
    )

    pops_ax = axes[0, 2]
    handles, labels = pops_ax.get_legend_handles_labels()
    order = [labels.index("Truth"), labels.index(r"$95.45\%$"), labels.index("max/min")]
    axes[0, 1].legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize=6,
        loc="lower center",
    )

    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    output = Path(__file__).resolve().parent / "example_polynomial.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
