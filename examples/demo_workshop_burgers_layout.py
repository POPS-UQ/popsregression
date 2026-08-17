"""Polynomial workshop figure in the Burgers-style row/column layout.

Rows are training-set sizes; columns are BayesianRidge, POPS Hypercube,
POPS Ellipse, and POPS Ellipse + PAC.  The visual convention matches the
Burgers example: an outer high-confidence/support band and a darker inner
band, with truth and training data overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from popsregression import POPSRegression, POPSRegressionEllipse


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


def draw_common(ax, x, truth, x_train, y_train, mean):
    ax.plot(x, truth, "k-", lw=1.6, label="Truth")
    ax.plot(x, mean, "C1-", lw=2.8, label="Mean")
    ax.plot(x_train, y_train, "b.", ms=4, label="Train")


def main():
    rng = np.random.RandomState(42)
    train_sizes = (10, 500)
    titles = ["Bayesian Ridge", "POPS Hypercube", "POPS Ellipse", "POPS Ellipse + PAC"]

    # Match the compact Burgers presentation: two rows, four method columns.
    fig, axes = plt.subplots(
        len(train_sizes), 4, figsize=(8, 4), sharex=True, sharey=True
    )

    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11)

    for row, n_samples in enumerate(train_sizes):
        X_train, x_train, y_train, X_dense, x_dense, y_dense = generate_data(
            rng, n_samples
        )

        bay = BayesianRidge(fit_intercept=False).fit(X_train, y_train)
        hyc = POPSRegression(
            minimum_relative_error=0.0, posterior="hypercube"
        ).fit(X_train, y_train)
        ell = POPSRegressionEllipse(random_state=0).fit(X_train, y_train)
        pac = POPSRegressionEllipse(random_state=0, pac_bayes=True).fit(
            X_train, y_train
        )

        # BayesianRidge: +/-4 sigma outer, +/-2 sigma inner.
        mean = bay.predict(X_dense)
        std = epistemic_std(bay, X_dense)
        ax = axes[row, 0]
        ax.fill_between(
            x_dense,
            mean - 4 * std,
            mean + 4 * std,
            alpha=0.20,
            facecolor="0.5",
            label=r"$99.997\%$ ($\pm4\sigma$)",
        )
        ax.fill_between(
            x_dense,
            mean - 2 * std,
            mean + 2 * std,
            alpha=0.50,
            facecolor="C1",
            label=r"$95.45\%$ ($\pm2\sigma$)",
        )
        draw_common(ax, x_dense, y_dense, x_train, y_train, mean)

        # POPS variants: full max/min outer band and +/-2 predictive std inner.
        for col, model in ((1, hyc), (2, ell), (3, pac)):
            mean, std, hi, lo = model.predict(
                X_dense, return_std=True, return_bounds=True
            )
            ax = axes[row, col]
            ax.fill_between(
                x_dense,
                lo,
                hi,
                alpha=0.20,
                facecolor="0.5",
                label="max/min",
            )
            ax.fill_between(
                x_dense,
                mean - 2 * std,
                mean + 2 * std,
                alpha=0.50,
                facecolor="C1",
                label=r"$95.45\%$ ($\pm2\sigma$)",
            )
            draw_common(ax, x_dense, y_dense, x_train, y_train, mean)

        axes[row, 0].set_ylabel(f"N = {n_samples}", fontsize=11)

    for ax in axes.flat:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-150, 150)
        ax.tick_params(labelsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("x")

    # Keep one unobtrusive legend, as in the pasted workshop figure.
    axes[1, 0].legend(fontsize=7, loc="lower right")

    fig.tight_layout(pad=0.7, w_pad=0.4, h_pad=0.3)
    fig.savefig("demo_workshop_burgers_layout.png", dpi=180, bbox_inches="tight")
    fig.savefig("demo_workshop_burgers_layout.pdf", bbox_inches="tight")
    print("Saved demo_workshop_burgers_layout.png and .pdf")


if __name__ == "__main__":
    main()
