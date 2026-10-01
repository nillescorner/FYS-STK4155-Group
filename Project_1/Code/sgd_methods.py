"""Stochastic gradient descent (part h)"""

import numpy as np

from gradient_descent_methods import gradient
from optimizer_methods import optimiser


def make_batches(n, batch_size, rng):
    """
    Shuffle the indices and split them into minibatches (Section 4.7)

        Params:
        n (int): total number of samples
        batch_size (int): how large the batches are
        rng (np.random.generator): randomly shuffles the indices

        Returns:
        batches (list of NDArray): list of index arrays
    """

    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def step_length(t, t0, t1):
    """
    The learning-rate schedule of Eq. (4.40).
    
        Params: 
        t (int): current ipdate step
        t0 (float): schedule numerator, controls initial step
        t1 (float): scedule offset, contreols how fast step size decays

        Returns:
        step length (float): learning rate to use at step t
    """

    return t0 / (t + t1)


def sgd(X, y, method="plain", n_epochs=50, batch_size=5, gamma=0.1, schedule=None,lmbda=0.0, seed=2026, theta0=None, **kw):
    """
    Minibatch stochastic gradient descent, Eq. (4.34), with any optimiser of optimiser_step.
    schedule=(t0, t1) replaces the constant gamma by Eq. (4.40).  Returns the iterate after every epoch.

        Params:
        X (NDArray, shape: (n, p)): Design matrix X
        y (NDArray, shape: (n, )): y
        method (str): "plain", "momentum", "adagrad", "rmsprop" or "adam"
        n_epoch (int): number of epochs, default: 50
        gamma (float): learning rate, default: 0.1, only used if schedule = None
        schedule (tuple(float, float) or None): for a decaying learning rate, if None then constant gamma is used instead, default: None
        lmbda (float): penalty, default: 0.0 (OLS)
        seed (int): seed to generate same set of numbers, default: 2026
        theta0 (NDArray, shape: (p, )): starting value of theta value, optional, default: None
        **kw: extra keyword arguments needed depending on method used

        Returns: 
        sgd (NDArray, shape: (n_epochs + 1, p)): the value of theta after every epoch
    """
    
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

    return np.array(history), t + 1
