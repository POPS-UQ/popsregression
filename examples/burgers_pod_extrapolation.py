"""Harder POD-ROM Burgers example with genuine basis extrapolation error.

This reuses the full workshop pipeline in ``burgers_pod_sim2science.py`` but
constructs the offline POD basis only from a smoother regime (higher viscosity,
earlier time).  The regression/test cases still span the full Burgers domain,
and the displayed slice is low-viscosity/late-time.  Thus rank-3 POD is no
longer nearly exact: the persistent error is a realistic reduced-basis
extrapolation/truncation error rather than an artificially tiny retained rank.
"""

import argparse
import numpy as np

import burgers_pod_sim2science as base


# Smooth regime used only for the offline POD snapshot library.
POD_NU_RANGE = (0.040, base.NU_RANGE[1])
POD_T_RANGE = (base.T_RANGE[0], 0.55)


def build_smooth_pod_basis(rank=3, n_basis_cases=48, seed=base.SEED + 1000):
    """POD basis trained only on higher-nu / earlier-time smooth snapshots."""
    rng = np.random.default_rng(seed)
    cases = np.column_stack(
        [
            rng.uniform(*POD_NU_RANGE, n_basis_cases),
            rng.uniform(*base.AMP_RANGE, n_basis_cases),
            rng.uniform(*POD_T_RANGE, n_basis_cases),
        ]
    )

    snapshots = []
    for nu, amp, time in cases:
        _, u = base.burgers_solution(nu, amp, time)
        snapshots.append(u)
    snapshots = np.asarray(snapshots)

    mean_field = snapshots.mean(axis=0)
    centered = snapshots - mean_field[None, :]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    modes = vt[:rank]
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return mean_field, modes, singular_values, energy


def main(rank=3, basis_cases=48):
    # ``base.run`` resolves build_pod_basis from its module globals, so swap in
    # the smooth-regime library while retaining the exact same fits, diagnostics,
    # figure layout, and two-legend convention as the original POD example.
    base.build_pod_basis = build_smooth_pod_basis
    print(
        "Offline POD library restricted to smooth regime: "
        f"nu in {POD_NU_RANGE}, t in {POD_T_RANGE}; "
        f"full regression/test domain remains nu in {base.NU_RANGE}, "
        f"t in {base.T_RANGE}."
    )
    return base.run(pod_rank=rank, n_basis_cases=basis_cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=3, choices=(2, 3, 4, 5))
    parser.add_argument("--basis-cases", type=int, default=48)
    args = parser.parse_args()
    main(rank=args.rank, basis_cases=args.basis_cases)
