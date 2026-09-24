"""Supplementary ablation: weight selection on a finite POPS dictionary.

The quartic surrogate of ``example_polynomial.py`` is run through the
finite-dictionary construction :class:`comparisons.pops_dictionary.POPSPACCertificate`
(see ``docs/glossary.md`` for the terms used here): an independent pilot sample
``D0`` fixes the dictionary of candidate ellipsoids and every protocol
constant, and an independent certification sample ``D1`` only chooses
categorical weights. On the same frozen dictionary the four weight rules
(fixed prior, best single candidate, POPS-dictionary predictive stacking
and PAC-selected Gibbs) are compared: this is the optional weight-selection
ablation on POPS candidates, not the standalone Bayesian stacking baseline
of ``example_repeated_splits.py``. The package's own PAC construction,
``POPSEllipseRegression(regularization='PAC')``, uses a continuous Gaussian
hyperposterior instead of a finite dictionary. The bare POPS ellipse and Ellipse+EB
fitted on the full budget ``D0 + D1`` are listed for their held-out
predictive performance only; the EB ``bound_`` is a diagnostic objective
and is not placed next to the PAC upper bound.

Two repeated-draw studies are run with paired seeds:

- ``fixed_pilot``: one pilot, repeated independent certification draws
  (the conditional experiment of the theorem);
- ``repeated_pilot``: pilot and certification both redrawn (whole-pipeline
  variability).

Every run writes tidy rows to ``generated/pops_dictionary_ablation.csv``,
per-unit candidate and mixture losses to
``generated/pops_dictionary_ablation_units.npz``, the markdown summary
tables ``generated/pops_dictionary_ablation.md`` (medians over repeats), and
the supplementary band figure ``pops_dictionary_ablation.png``
(Ellipse+PAC, finite-dictionary Gibbs and POPS-dictionary stacking on the
data of the main polynomial figure). Seeds are predeclared (0, 1, 2, ...); the
tables aggregate over all repeats and never select a favorable one.

Output support statement. On the input domain ``|x| <= 10`` the target
``0.1 (x^3 + 0.01 x^4) + 10 x sin x`` satisfies ``|y| <= 100 + 10 + 100 =
210``; ``y_bounds = (-210, 210)`` is therefore a population statement, not a
sample extremum. The simulator is deterministic, so bounded outputs hold
exactly. Coverage and width of the exact 95.45% central interval of the
contaminated mixture are reported as descriptive metrics, not as certified
quantities: the certificate concerns the population log-risk only.
"""

import argparse
import csv
import time
from pathlib import Path

import example_polynomial as poly
import matplotlib.pyplot as plt
import numpy as np
from comparisons import harness
from comparisons.plotting import band_panel, legend_handles
from comparisons.pops_dictionary import POPSPACCertificate
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from popsregression import POPSEllipseRegression, POPSRegression
from popsregression._projected_ball import (
    floor_contaminated_logpdf,
    projected_ball_logpdf,
)

HERE = Path(__file__).resolve().parent
X_RANGE = 10.0
Y_BOUNDS = (-210.0, 210.0)  # analytic population support, see module docstring
DEGREE = 4

# Protocol constants, fixed before any certification data is seen.
PROTOCOL = dict(
    candidate_scales=np.geomspace(0.5, 4.0, 17),
    beta=0.02,
    y_bounds=Y_BOUNDS,
    min_half_width=0.5,
    lambdas=np.geomspace(0.25, 128.0, 24),
    failure_probability=0.05,
    constant_feature=0,  # the bias column of PolynomialFeatures
)
RULES = ("fixed_prior", "single_best", "stacking", "gibbs")
RULE_LABELS = {
    "fixed_prior": "fixed prior $q_0$",
    "single_best": "single best candidate",
    "stacking": "predictive stacking",
    "gibbs": "PAC Gibbs",
}


def target_function(x):
    return (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0


def features(x):
    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
    return poly.fit_transform(np.asarray(x).reshape(-1, 1))


def draw(rng, n):
    x = rng.uniform(-X_RANGE, X_RANGE, n)
    return features(x), target_function(x)


def coverage_and_width(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def bare_central_interval(model, X, level):
    """Exact central interval of the bare projected-ball pushforward."""
    mean, hi, lo = model.predict(X, return_bounds=True)
    a = 0.5 * (model._ball_dim + 1.0)
    frac = 2.0 * beta_dist.ppf(0.5 + 0.5 * level, a, a) - 1.0
    half = frac * 0.5 * (hi - lo)
    return mean - half, mean + half


def run_one(study, repeat, n_pilot, n_cert, n_test, level, unit_store):
    pilot_seed = 0 if study == "fixed_pilot" else 1000 + repeat
    cert_seed = 2000 + repeat
    rng_pilot = np.random.RandomState(pilot_seed)
    rng_cert = np.random.RandomState(cert_seed)
    rng_test = np.random.RandomState(3000 + repeat)
    X0, y0 = draw(rng_pilot, n_pilot)
    X1, y1 = draw(rng_cert, n_cert)
    X_test, y_test = draw(rng_test, n_test)

    row = dict(
        study=study,
        repeat=repeat,
        pilot_seed=pilot_seed,
        cert_seed=cert_seed,
        N0=n_pilot,
        N1=n_cert,
        N_total=n_pilot + n_cert,
        N_test=n_test,
        P=X0.shape[1],
        K=PROTOCOL["candidate_scales"].size,
        beta=PROTOCOL["beta"],
        min_half_width=PROTOCOL["min_half_width"],
        xi=PROTOCOL["failure_probability"],
        n_lambdas=PROTOCOL["lambdas"].size,
        y_lower=Y_BOUNDS[0],
        y_upper=Y_BOUNDS[1],
    )

    # --- Pilot-split certificate on the frozen dictionary ---
    t0 = time.perf_counter()
    certifier = POPSPACCertificate(random_state=pilot_seed, **PROTOCOL)
    certifier.fit_pilot(X0, y0)
    row["time_pilot"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = certifier.certify(X1, y1)
    row["time_certify"] = time.perf_counter() - t0
    row.update(
        certified=result.certified,
        loss_lower=result.loss_lower,
        loss_upper=result.loss_upper,
        loss_range=result.loss_range,
        n_independent_units=result.n_independent_units,
        pilot_coverage=certifier.pilot_.pilot_coverage_fraction,
    )
    log_p_test = certifier.candidate_log_density(X_test, y_test)
    key = f"{study}_r{repeat}_N{n_cert}"
    unit_store[f"{key}_unit_losses"] = result.unit_losses
    unit_store[f"{key}_candidate_scales"] = PROTOCOL["candidate_scales"]
    unmod = result.loss_summary["unmodified_row_loss"]
    for name in RULES:
        rule = result.rules[name]
        q = rule.weights
        with np.errstate(divide="ignore"):
            test_losses = -np.log(np.exp(log_p_test) @ q)
        lo, hi = certifier.predict_interval(X_test, level=level, weights=q)
        cov, width = coverage_and_width(y_test, lo, hi)
        k_best = int(np.argmax(q))
        row.update(
            {
                f"{name}_empirical": rule.empirical,
                f"{name}_mixture_empirical": rule.mixture_empirical,
                f"{name}_jensen_gap": rule.jensen_gap,
                f"{name}_kl": rule.kl,
                f"{name}_complexity_penalty": rule.complexity_penalty,
                f"{name}_hoeffding_penalty": rule.concentration_penalty,
                f"{name}_raw_bound": rule.raw_bound,
                f"{name}_capped_bound": rule.capped_bound,
                f"{name}_nonvacuous": rule.is_nonvacuous,
                f"{name}_selected_lambda": rule.selected_lambda,
                f"{name}_test_mixture_nll": float(test_losses.mean()),
                f"{name}_test_mixture_nll_se": float(
                    test_losses.std(ddof=1) / np.sqrt(test_losses.size)
                ),
                f"{name}_test_coverage": cov,
                f"{name}_test_width": width,
                f"{name}_floor_fraction": rule.floor_fraction,
                f"{name}_unit_loss_max": float(rule.unit_mixture_losses.max()),
                f"{name}_unit_loss_q99": float(
                    np.quantile(rule.unit_mixture_losses, 0.99)
                ),
                f"{name}_unit_loss_var": float(rule.unit_mixture_losses.var()),
                f"{name}_top_scale": float(PROTOCOL["candidate_scales"][k_best]),
                f"{name}_top_weight": float(q[k_best]),
                f"{name}_n_nonzero_weights": int(np.count_nonzero(q)),
                f"{name}_top_candidate_outside_support_fraction": float(
                    result.fraction_outside_support[k_best]
                ),
                f"{name}_top_candidate_unmodified_nonfinite": int(
                    unmod["n_nonfinite"][k_best]
                ),
            }
        )
        unit_store[f"{key}_{name}_unit_mixture_losses"] = rule.unit_mixture_losses
    row["stacking_solver"] = result.rules["stacking"].solver_status["method"]
    row["stacking_success"] = result.rules["stacking"].solver_status["success"]

    # --- One-sample methods on the full budget D0 + D1 ---
    X_all = np.vstack([X0, X1])
    y_all = np.concatenate([y0, y1])
    t0 = time.perf_counter()
    bare = POPSEllipseRegression(random_state=0).fit(X_all, y_all)
    row["time_bare"] = time.perf_counter() - t0
    mean, hi, lo = bare.predict(X_test, return_bounds=True)
    ball_dim = bare._ball_dim
    half = 0.5 * (hi - lo)
    exact_loss = -floor_contaminated_logpdf(
        y_test - mean,
        half,
        ball_dim,
        beta=PROTOCOL["beta"],
        y=y_test,
        y_bounds=Y_BOUNDS,
        strict=False,
    )
    # Unmodified compact-support loss: may be infinite; never capped.
    unmodified = -projected_ball_logpdf(y_test - mean, half, ball_dim)
    finite = np.isfinite(unmodified)
    cov_sup, width_sup = coverage_and_width(y_test, lo, hi)
    q_lo, q_hi = bare_central_interval(bare, X_test, level)
    cov_q, width_q = coverage_and_width(y_test, q_lo, q_hi)
    row.update(
        bare_test_nll_contaminated=float(exact_loss.mean()),
        bare_test_unmodified_nonfinite_fraction=float(1.0 - finite.mean()),
        bare_test_unmodified_nll_finite_part=(
            float(unmodified[finite].mean()) if finite.any() else np.nan
        ),
        bare_test_support_coverage=cov_sup,
        bare_test_support_width=width_sup,
        bare_test_coverage=cov_q,
        bare_test_width=width_q,
    )

    t0 = time.perf_counter()
    eb = POPSEllipseRegression(regularization="empirical-bayes", random_state=0)
    eb.fit(X_all, y_all)
    row["time_eb"] = time.perf_counter() - t0
    cov_out, width_out = coverage_and_width(
        y_test, *eb.predict_interval(X_test, level=0.999)
    )
    cov_in, width_in = coverage_and_width(
        y_test, *eb.predict_interval(X_test, level=level)
    )
    row.update(
        eb_diagnostic_bound=eb.diagnostic_bound_,
        eb_kl=eb.kl_,
        eb_certificate_status=eb.certificate_status_,
        eb_test_outer_coverage=cov_out,
        eb_test_outer_width=width_out,
        eb_test_coverage=cov_in,
        eb_test_width=width_in,
    )

    hyc = POPSRegression(minimum_relative_error=0.0, random_state=0).fit(X_all, y_all)
    _, hi, lo = hyc.predict(X_test, return_bounds=True)
    row["hypercube_test_coverage"], row["hypercube_test_width"] = coverage_and_width(
        y_test, lo, hi
    )

    bay = BayesianRidge(fit_intercept=False).fit(X_all, y_all)
    mean = bay.predict(X_test)
    epi = np.sqrt(np.sum((X_test @ bay.sigma_) * X_test, axis=1))
    full = np.sqrt(epi**2 + 1.0 / bay.alpha_)
    row["bayes_epistemic_2std_coverage"], row["bayes_epistemic_2std_width"] = (
        coverage_and_width(y_test, mean - 2 * epi, mean + 2 * epi)
    )
    row["bayes_predictive_2std_coverage"], row["bayes_predictive_2std_width"] = (
        coverage_and_width(y_test, mean - 2 * full, mean + 2 * full)
    )
    row["bayes_predictive_test_nll"] = float(
        np.mean(
            0.5 * np.log(2 * np.pi * full**2) + 0.5 * ((y_test - mean) / full) ** 2
        )
    )
    return row


def _agg(rows, study, n, key, fn=np.median):
    return fn([float(r[key]) for r in rows if r["study"] == study and r["N1"] == n])


def markdown_table(rows, study, sizes):
    """Median over repeats of every reported quantity, one row per rule."""
    lines = [
        (
            f"Study `{study}`: medians over repeats; nats unless stated. "
            "`empirical` is the PAC term sum_k q_k g_k, `mixture` the mixture log "
            "loss on D1, `Jensen` their difference, `KL`/`complexity`/`Hoeffding` "
            "the bound terms at the selected temperature, `raw` the complete PAC "
            "right side, `ceiling` the trivial bound log(R_y/beta), `non-vac.` the "
            "fraction of repeats with raw < ceiling, `test NLL` the mixture log loss "
            "on an independent test sample (with its standard error), `floor` the "
            "mean fraction of predictive density supplied by the uniform floor, and "
            "`cov`/`width` the descriptive coverage and mean width of the exact "
            "95.45% central interval on the test sample."
        ),
        "",
        (
            "| N1 | rule | empirical | mixture | Jensen | KL | complexity | Hoeffding"
            " | raw | ceiling | non-vac. | test NLL | floor | cov | width |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in sizes:
        for name in RULES:
            g = lambda k: _agg(rows, study, n, f"{name}_{k}")  # noqa: E731
            nonvac = np.mean(
                [
                    r[f"{name}_nonvacuous"] in (True, "True")
                    for r in rows
                    if r["study"] == study and r["N1"] == n
                ]
            )
            lines.append(
                f"| {n} | {RULE_LABELS[name]} | {g('empirical'):.2f}"
                f" | {g('mixture_empirical'):.2f} | {g('jensen_gap'):.3f}"
                f" | {g('kl'):.2f} | {g('complexity_penalty'):.2f}"
                f" | {g('hoeffding_penalty'):.2f} | **{g('raw_bound'):.2f}**"
                f" | {_agg(rows, study, n, 'loss_upper'):.2f} | {100 * nonvac:.0f}%"
                f" | {g('test_mixture_nll'):.3f} ± {g('test_mixture_nll_se'):.3f}"
                f" | {g('floor_fraction'):.3f} | {g('test_coverage'):.3f}"
                f" | {g('test_width'):.0f} |"
            )
        e = lambda k, fn=np.median: _agg(rows, study, n, k, fn)  # noqa: E731
        lines.append(
            f"| {n} | Ellipse+EB (one sample, N0+N1) | – | – | – | – | – | – | –"
            " (diagnostic objective, not a PAC bound) | – | – | – | – |"
            f" {e('eb_test_coverage'):.3f}"
            f" | {e('eb_test_width'):.0f} |"
        )
        lines.append(
            f"| {n} | bare ellipse (one sample, N0+N1) | – | – | – | – | – | – | – |"
            f" – | – | {e('bare_test_nll_contaminated'):.3f} (contaminated),"
            f" {e('bare_test_unmodified_nll_finite_part'):.3f} (unmodified,"
            f" {100 * e('bare_test_unmodified_nonfinite_fraction', np.mean):.1f}%"
            f" non-finite) | – | {e('bare_test_coverage'):.3f}"
            f" | {e('bare_test_width'):.0f} |"
        )
    return "\n".join(lines)


def _parse(value):
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def aggregate_mean(rows, study, key):
    sizes = sorted({r["N1"] for r in rows if r["study"] == study})
    return [
        np.mean([float(r[key]) for r in rows if r["study"] == study and r["N1"] == n])
        for n in sizes
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--pilot-size", type=int, default=40)
    parser.add_argument("--cert-sizes", type=int, nargs="+", default=[10, 30, 100, 300])
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--level", type=float, default=0.9545)
    parser.add_argument(
        "--studies", nargs="+", default=["fixed_pilot", "repeated_pilot"]
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="only rebuild the tables from the saved CSV (no computation)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pops_dictionary_ablation.csv"
    if args.from_csv:
        with open(csv_path, newline="") as fh:
            rows = [
                {k: _parse(v) for k, v in row.items()} for row in csv.DictReader(fh)
            ]
        write_tables(rows, args)
        return

    rows, units = [], {}
    for study in args.studies:
        for repeat in range(args.repeats):
            for n_cert in args.cert_sizes:
                rows.append(
                    run_one(
                        study,
                        repeat,
                        args.pilot_size,
                        n_cert,
                        args.test_size,
                        args.level,
                        units,
                    )
                )
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "pops_dictionary_ablation_units.npz", **units)
    print(f"Saved {csv_path} ({len(rows)} rows)")

    write_tables(rows, args)


def supplement_figure(output, level=0.9545):
    """Ellipse+PAC, finite-dictionary Gibbs and POPS-dictionary stacking on
    the data of the main polynomial figure (same seed and sizes). All bands
    are exact central intervals of the parameter-only predictive (95.45%
    inner, 99.9% outer), as in the main figures."""
    rng = np.random.RandomState(poly.SEED)
    titles = ["Ellipse+PAC", "Finite dictionary: Gibbs", "Finite dictionary: stacking"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for row, n_samples in enumerate(poly.TRAIN_SIZES):
        problem, x_train, x_dense = poly.make_problem(rng, n_samples)
        X_train, y_train = problem.X_train, problem.y_train
        # Same certified loss as the dictionary: identical y_bounds, floor
        # weight and minimum half-width, so all three bounds are comparable.
        pac = POPSEllipseRegression(
            regularization="PAC",
            y_bounds=PROTOCOL["y_bounds"],
            floor_weight=PROTOCOL["beta"],
            min_half_width=PROTOCOL["min_half_width"],
            random_state=poly.SEED,
        ).fit(X_train, y_train)
        pred = harness.Predictive(
            "Ellipse+PAC",
            pac.predict(problem.X_test),
            {lv: pac.predict_interval(problem.X_test, lv) for lv in (level, 0.999)},
            info={"bound": pac.bound_, "trivial_bound": pac.certificate_.trivial_bound},
        )
        band_panel(axes[row, 0], x_dense, pred, problem.y_test, x_train, y_train)
        labels = [
            f"bound {pred.info['bound']:.1f} (ceiling {pred.info['trivial_bound']:.1f})"
        ]

        # Finite dictionary: a random half fixes the dictionary, the other
        # half only chooses the categorical weights. Nothing is refitted.
        perm = rng.permutation(len(y_train))
        pilot, cert = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        certifier = POPSPACCertificate(random_state=0, **PROTOCOL)
        certifier.fit_pilot(X_train[pilot], y_train[pilot])
        result = certifier.certify(X_train[cert], y_train[cert])
        for col, key in ((1, "gibbs"), (2, "stacking")):
            rule = result.rules[key]
            intervals = {
                lv: certifier.predict_interval(
                    problem.X_test, level=lv, weights=rule.weights, contaminated=False
                )
                for lv in (level, 0.999)
            }
            dict_pred = harness.Predictive(
                key, certifier.predict(problem.X_test, weights=rule.weights), intervals
            )
            ax = axes[row, col]
            band_panel(ax, x_dense, dict_pred, problem.y_test, x_train, y_train)
            ax.plot(x_train[pilot], y_train[pilot], "g.", ms=3.5)
            labels.append(
                f"bound {rule.raw_bound:.1f} (ceiling {result.loss_upper:.1f})"
            )
        for col, text in enumerate(labels):
            axes[row, col].text(
                0.5,
                1.015,
                text,
                transform=axes[row, col].transAxes,
                fontsize=6.5,
                ha="center",
                va="bottom",
            )
        axes[row, 0].set_ylabel(f"N = {n_samples}", fontsize=9)
        print(
            f"N={n_samples}: N0={pilot.size}, N1={cert.size}; Ellipse+PAC bound"
            f" {pred.info['bound']:.2f}; dictionary Gibbs bound {result.raw_bound:.2f},"
            f" dictionary stacking bound {result.rules['stacking'].raw_bound:.2f}"
        )
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9, pad=12)
    for ax in axes.flat:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-150, 150)
        ax.tick_params(labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("x", fontsize=9)
    handles = legend_handles("certification half") + [
        Line2D([], [], color="g", marker=".", ls="", ms=5, label="pilot half")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(pad=0.2, w_pad=0.25, h_pad=0.9, rect=(0, 0.12, 1, 1))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def write_tables(rows, args):
    tables = []
    for study in args.studies:
        tables.append(markdown_table(rows, study, args.cert_sizes))
    text = "\n\n".join(tables) + "\n"
    md_path = args.output_dir / "pops_dictionary_ablation.md"
    md_path.write_text(text)
    print(text)
    print(f"Saved {md_path}")
    supplement_figure(HERE / "pops_dictionary_ablation.png", args.level)


if __name__ == "__main__":
    main()
