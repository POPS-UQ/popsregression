"""Polynomial workshop figure matching the Burgers confidence-band presentation."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from popsregression import POPSRegression, POPSRegressionEllipse


def target_function(x):
    return (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0


def generate_data(N):
    x_train = np.sort(
        np.append(np.random.uniform(-1, 1, N), np.linspace(-1, 1, 2)) * 10
    )
    x_dense = np.linspace(-1.1, 1.1, 101) * 10
    y_dense = target_function(x_dense)
    poly = PolynomialFeatures(degree=4, include_bias=True)
    X_train = poly.fit_transform(x_train.reshape(-1, 1))
    X_dense = poly.transform(x_dense.reshape(-1, 1))
    y_train = target_function(x_train)
    return X_train, x_train, y_train, X_dense, x_dense, y_dense


def plot_panel(ax, x_dense, y_dense, x_train, y_train, y_pred, y_std,
               y_max=None, y_min=None):
    if y_max is None:
        ax.fill_between(
            x_dense, y_pred - 4*y_std, y_pred + 4*y_std,
            alpha=0.20, facecolor="0.5",
            label=r"$99.997\%$ confidence ($\pm4\sigma$)",
        )
        ax.fill_between(
            x_dense, y_pred - 2*y_std, y_pred + 2*y_std,
            alpha=0.45, facecolor="C1",
            label=r"$95.45\%$ confidence ($\pm2\sigma$)",
        )
    else:
        # Keep the same visual convention as the Burgers workshop figure:
        # the outer posterior range is presented as the high-confidence band,
        # while +/-2 predictive std is the inner 95.45% band.
        ax.fill_between(
            x_dense, y_min, y_max, alpha=0.20, facecolor="0.5",
            label=r"$99.997\%$ confidence",
        )
        ax.fill_between(
            x_dense, y_pred - 2*y_std, y_pred + 2*y_std,
            alpha=0.45, facecolor="C1",
            label=r"$95.45\%$ confidence",
        )
    ax.plot(x_dense, y_pred, "C1-", lw=3)
    ax.plot(x_train, y_train, "b.", ms=4, label="Train")
    ax.plot(x_dense, y_dense, "k-", lw=1.2, label="Truth")


def main():
    np.random.seed(42)
    titles = ["Bayesian Ridge", "POPS Hypercube", "POPS Ellipse",
              "POPS Ellipse + PAC"]
    fig, axs = plt.subplots(4, 3, figsize=(8, 9.5), sharex=True, sharey=True)
    N_array = [10, 50, 500]

    for i, N in enumerate(N_array):
        X_train, x_train, y_train, X_dense, x_dense, y_dense = generate_data(N)
        bay = BayesianRidge(fit_intercept=False)
        hyc = POPSRegression(minimum_relative_error=0.0, posterior="hypercube")
        ell = POPSRegressionEllipse(random_state=0)
        pac = POPSRegressionEllipse(random_state=0, pac_bayes=True)
        bay.fit(X_train, y_train)
        hyc.fit(X_train, y_train)
        ell.fit(X_train, y_train)
        pac.fit(X_train, y_train)

        b_pred = bay.predict(X_dense)
        b_std = np.sqrt(np.sum((X_dense @ bay.sigma_) * X_dense, axis=1))
        plot_panel(axs[0, i], x_dense, y_dense, x_train, y_train, b_pred, b_std)

        for row, model in ((1, hyc), (2, ell), (3, pac)):
            y_pred, y_std, y_max, y_min = model.predict(
                X_dense, return_std=True, return_bounds=True
            )
            plot_panel(axs[row, i], x_dense, y_dense, x_train, y_train,
                       y_pred, y_std, y_max=y_max, y_min=y_min)

        axs[0, i].set_title(f"N = {N}")

    for j, title in enumerate(titles):
        axs[j, 0].set_ylim(-250, 250)
        axs[j, 0].set_xlim(-10, 10)
        axs[j, 1].legend(fontsize=7, loc="lower center")
        axs[j, 0].set_ylabel(title, fontsize=11)
    for ax in axs[-1]:
        ax.set_xlabel("x")

    plt.tight_layout()
    fig.savefig("demo_workshop.png", dpi=180, bbox_inches="tight")
    fig.savefig("demo_workshop.pdf", bbox_inches="tight")
    print("Saved demo_workshop.png and demo_workshop.pdf")


if __name__ == "__main__":
    main()
