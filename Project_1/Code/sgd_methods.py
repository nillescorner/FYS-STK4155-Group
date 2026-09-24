"""Stochastic gradient descent (part h)"""

import numpy as np

from gradient_descent_methods import gradient
from optimizer_methods import optimiser


def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches (Section 4.7)."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def step_length(t, t0, t1):
    """The learning-rate schedule of Eq. (4.40)."""
    return t0 / (t + t1)


def sgd(X, y, method="plain", n_epochs=50, batch_size=5, gamma=0.1, schedule=None,lmbda=0.0, seed=2026, theta0=None, **kw):
    """Minibatch stochastic gradient descent, Eq. (4.34), with any optimiser of optimiser_step.
    schedule=(t0, t1) replaces the constant gamma by Eq. (4.40).  Returns the iterate after every epoch."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else np.array(theta0, dtype=float)
    state, t, history = {}, 0, [theta.copy()]
    for epoch in range(n_epochs):
        for batch in make_batches(n, batch_size, rng):
            t += 1
            g = gradient(theta, X[batch], y[batch], lmbda)            # Eq. (4.33): the minibatch gradient
            g_t = gamma if schedule is None else step_length(t, *schedule)
            theta, state = optimiser(method, theta, g, state, t, g_t, **kw)
        history.append(theta.copy())
    return np.array(history)
