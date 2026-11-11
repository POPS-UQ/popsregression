"""
Limited-memory BFGS in JAX.

A compact L-BFGS with an Armijo backtracking line search, written with
``jax.lax`` control flow so that a whole minimization (and a continuation
over several of them) compiles to one kernel and can be batched with
``jax.vmap``. The stopping rule is that of SciPy's ``L-BFGS-B`` with
``ftol = gtol = tol``.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import jax
import jax.numpy as jnp

_ARMIJO = 1e-4
_MAX_BACKTRACKS = 30
# A trial step may be at most this many times longer than the last accepted
# one. The barrier objectives of this package fall steeply towards a nearly
# flat plateau far from the minimum; an unbounded quasi-Newton step can land
# on it, where the gradient is too small to come back.
_MAX_STEP_GROWTH = 4.0


class _State(NamedTuple):
    k: jnp.ndarray
    x: jnp.ndarray
    f: jnp.ndarray
    g: jnp.ndarray
    S: jnp.ndarray
    Y: jnp.ndarray
    R: jnp.ndarray
    head: jnp.ndarray
    last_step: jnp.ndarray
    done: jnp.ndarray


def _direction(g, S, Y, R, head):
    """Two-loop recursion; empty history slots have ``R = 0`` and drop out."""
    m = S.shape[0]
    order = [(head - 1 - j) % m for j in range(m)]
    q = g
    alphas = []
    for i in order:
        a = R[i] * (S[i] @ q)
        q = q - a * Y[i]
        alphas.append(a)
    newest = order[0]
    yy = Y[newest] @ Y[newest]
    gamma = jnp.where(
        R[newest] > 0, 1.0 / jnp.where(R[newest] > 0, R[newest] * yy, 1.0), 1.0
    )
    z = gamma * q
    for i, a in zip(reversed(order), reversed(alphas)):
        z = z + S[i] * (a - R[i] * (Y[i] @ z))
    return -z


def lbfgs(value_and_grad, x0, *, max_iter, tol, history=10):
    """Minimize a smooth function.

    Parameters
    ----------
    value_and_grad : callable
        ``x -> (value, gradient)``.
    x0 : array of shape (n,)
        Starting point.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Stop when the largest gradient entry is below ``tol`` or the
        relative decrease of the value is below ``tol``.
    history : int, default=10
        Number of stored curvature pairs.

    Returns
    -------
    x : array of shape (n,)
    value : float
    n_iter : int
    """
    f0, g0 = value_and_grad(x0)
    n = x0.shape[0]
    zeros = jnp.zeros((history, n), x0.dtype)
    start = _State(
        k=jnp.asarray(0),
        x=x0,
        f=f0,
        g=g0,
        S=zeros,
        Y=zeros,
        R=jnp.zeros(history, x0.dtype),
        head=jnp.asarray(0),
        last_step=jnp.asarray(jnp.inf, x0.dtype),
        done=jnp.max(jnp.abs(g0)) <= tol,
    )

    def step(state):
        d = _direction(state.g, state.S, state.Y, state.R, state.head)
        slope = state.g @ d
        # A direction that is not a descent direction means the stored
        # curvature is stale: fall back to steepest descent.
        descent = slope < 0
        d = jnp.where(descent, d, -state.g)
        slope = jnp.where(descent, slope, -(state.g @ state.g))
        fresh = jnp.all(state.R == 0) | ~descent
        longest = _MAX_STEP_GROWTH * state.last_step / jnp.linalg.norm(d)
        t0 = jnp.where(
            fresh,
            jnp.minimum(1.0, 1.0 / jnp.linalg.norm(state.g)),
            jnp.minimum(1.0, longest),
        )

        def trial(t):
            f, g = value_and_grad(state.x + t * d)
            return f, g

        def insufficient(ls):
            t, f, _, j = ls
            failed = ~jnp.isfinite(f) | (f > state.f + _ARMIJO * t * slope)
            return failed & (j < _MAX_BACKTRACKS)

        def backtrack(ls):
            t, f, _, j = ls
            # Minimizer of the quadratic through f(0), f'(0) and f(t).
            t_quad = -0.5 * slope * t * t / (f - state.f - slope * t)
            t_new = jnp.where(
                jnp.isfinite(t_quad), jnp.clip(t_quad, 0.1 * t, 0.5 * t), 0.5 * t
            )
            f_new, g_new = trial(t_new)
            return t_new, f_new, g_new, j + 1

        f1, g1 = trial(t0)
        t, f_new, g_new, _ = jax.lax.while_loop(
            insufficient, backtrack, (t0, f1, g1, 0)
        )
        accepted = jnp.isfinite(f_new) & (f_new <= state.f + _ARMIJO * t * slope)

        s = t * d
        y = g_new - state.g
        sy = s @ y
        curved = accepted & (sy > 1e-10 * jnp.linalg.norm(s) * jnp.linalg.norm(y))
        head = state.head
        S = jnp.where(curved, state.S.at[head].set(s), state.S)
        Y = jnp.where(curved, state.Y.at[head].set(y), state.Y)
        R = jnp.where(
            curved, state.R.at[head].set(1.0 / jnp.where(curved, sy, 1.0)), state.R
        )
        head = jnp.where(curved, (head + 1) % history, head)
        # After a failed search from a quasi-Newton direction, retry once
        # from steepest descent before giving up.
        retry = ~accepted & ~fresh
        R = jnp.where(retry, jnp.zeros_like(R), R)

        scale = jnp.maximum(jnp.maximum(jnp.abs(state.f), jnp.abs(f_new)), 1.0)
        converged = (jnp.max(jnp.abs(g_new)) <= tol) | (state.f - f_new <= tol * scale)
        done = jnp.where(accepted, converged, ~retry)
        return _State(
            k=state.k + 1,
            x=jnp.where(accepted, state.x + s, state.x),
            f=jnp.where(accepted, f_new, state.f),
            g=jnp.where(accepted, g_new, state.g),
            S=S,
            Y=Y,
            R=R,
            head=head,
            last_step=jnp.where(accepted, jnp.linalg.norm(s), state.last_step),
            done=done,
        )

    def running(state):
        return ~state.done & (state.k < max_iter)

    final = jax.lax.while_loop(running, step, start)
    return final.x, final.f, final.k
