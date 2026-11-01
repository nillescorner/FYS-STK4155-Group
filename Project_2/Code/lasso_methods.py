"""Lasso regression methods (gradient descent w/ soft threshold, coordinate descent)"""

import numpy as np

def soft_threshold(z, gamma):
    """
    Need a soft threshold for the l1 penalty since we cant differentiate at theta = 0

        Params:
        z (NDArray): value to give a soft threshold
        gamma(float): threshold, shrinkage amount

        Returns:
        soft threshold (NDArray, same shape as z): the soft threshold values 

    """

    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

def lasso_coordinate_descent(X, y, lmbda, n_iter=10000, tol=1e-8):
    """
    Lasso by cyclic coordinate descent.

    Minimises ||y - X theta||^2 / n + lmbda * ||theta||_1.
    The columns of X are assumed centred and standardised, and no
    intercept is penalised

        Params: 
        X (NDArray, shape: (n, p)): Design matrix X
        y (NDArray, shape: (n, )): y
        lmbda (flaot): penalty lambda, set to 0.0 for OLS, choose a different value for Ridge or Lasso
        num_iters (int): number of iterations, default: 10000
        tol (float): tolerence value, code stops if the gradient is under the tolerence, default: 1e-8

        Returns: 
        history (NDArray, shape: (n_steps + 1, p)): the theta value at every step
        n_steps (int): number of iterations taken
    """

    n, p = X.shape
    theta = np.zeros(p)
    col_norms = np.sum(X**2, axis=0)
    r = y - X @ theta                              #full residual
    history = [theta.copy()]

    for t in range(n_iter):
        theta_old = theta.copy()
        for j in range(p):
            #partial residual: add back the current contribution of column j
            r += X[:, j] * theta[j]
            rho = X[:, j] @ r
            theta[j] = soft_threshold(rho, lmbda * n / 2.0) / col_norms[j]
            r -= X[:, j] * theta[j]                #remove the updated one

        history.append(theta.copy())
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return history, t + 1
