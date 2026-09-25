"""Display names and caption text shared by every figure and table.

Internal method identifiers (CSV ``method`` column, command-line
``--methods``) are unchanged; only what a reader sees goes through
:func:`display`.
"""

DISPLAY_NAMES = {
    "POPS ellipse": "POPS Ellipse",
    "Ellipse+EB": "POPS Ellipse+EB",
    "Ellipse+PAC": "POPS Ellipse+PAC",
    "POPS ellipse (free center)": "POPS Ellipse (free center)",
    "Ellipse+EB (free center)": "POPS Ellipse+EB (free center)",
    "Ellipse+PAC (free center)": "POPS Ellipse+PAC (free center)",
    "POPS hypercube": "POPS hypercube",
    "PAC2-T": "PAC$^2_T$",
}

# Expansions of every abbreviation used in a caption, header or legend.
ABBREVIATIONS = {
    "POPS": "pointwise optimal parameter sets",
    "EB": "empirical Bayes",
    "PAC": "probably approximately correct (PAC-Bayes hyperparameter bound)",
    "PACm": "PAC-Bayes objective with an m-sample predictive (Morningstar et al. 2022)",
    "PAC$^2_T$": (
        "second-order PAC-Bayes objective with the tandem Taylor weight (Masegosa 2020)"
    ),
    "PVI": "predictive variational inference (Lai, Linero and Yao)",
    "IS": "interval score (Gneiting and Raftery), of the 95.45% central interval",
    "NLL": "negative log predictive density",
    "RMSE": "root-mean-square error of the predictive mean",
    "A_abs": "unsigned calibration area, int |C(u) - u| du",
    "A_s": "signed calibration area, int (C(u) - u) du",
    "N/P": "training structures per model parameter",
    "ACE": "atomic cluster expansion",
}

DISSIPATION_DEFINITION = (
    "Burgers dissipation Q = nu * int |du/dx|^2 dx, the rate at which "
    "viscosity removes kinetic energy from the solution u(x, t)."
)


def display(method):
    """Reader-facing name of an internal method identifier."""
    return DISPLAY_NAMES.get(method, method)


def abbreviation_note(keys):
    """One caption sentence expanding the given abbreviations, in order."""
    return "; ".join(f"{k}: {ABBREVIATIONS[k]}" for k in keys) + "."
