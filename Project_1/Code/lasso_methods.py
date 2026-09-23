"""Lasso regression methods (gradient descent w/ soft threshold, coordinate descent)"""

import numpy as np

#need a soft threshold for the l1 penalty since we cant differentiate at theta = 0
def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

"""lasso gradient descent"""
def lasso_gd(X, y, gamma, lmbda = 0.0, n_iters = 40000, tol = 1e-8, theta0 = None):
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else theta0.copy()
    history = [theta.copy()]

    for t in range(n_iters):
        smooth_gradient = (2.0 / n) * X.T @ (X @ theta - y)

        theta_new = theta - gamma * smooth_gradient
        theta_new = soft_threshold(theta_new, gamma * lmbda)

        history.append(theta_new.copy())
        
        if np.linalg.norm(theta_new - theta)< tol:
            theta = theta_new
            break

        theta = theta_new

    return history, t + 1


"""Coordinate descent"""
def lasso_coordinate_descent(X, y, lmbda, n_iter=1000, tol=1e-8):
    """Lasso by cyclic coordinate descent.

    Minimises ||y - X theta||^2 / n + lmbda * ||theta||_1.
    The columns of X are assumed centred and standardised, and no
    intercept is penalised
    """
    n, p = X.shape
    theta = np.zeros(p)
    col_norms = np.sum(X**2, axis=0)
    r = y - X @ theta                              # full residual
    history = [theta.copy()]

    for t in range(n_iter):
        theta_old = theta.copy()
        for j in range(p):
            # partial residual: add back the current contribution of column j
            r += X[:, j] * theta[j]
            rho = X[:, j] @ r
            theta[j] = soft_threshold(rho, lmbda * n / 2.0) / col_norms[j]
            r -= X[:, j] * theta[j]                # remove the updated one

        history.append(theta.copy())
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return history, t + 1


def cost_plot(history, X, y, lmbda = 0.0, penalty = "L2"):
    n = X.shape[0]
    costs = []

    for theta in history:
        mse = (1.0 / n) * np.sum((X @ theta - y) **2)

        #penalty for ridge and lasso
        if penalty == "L2":
            reg = lmbda * np.sum(theta **2)
        elif penalty =="L1":
            reg = lmbda * np.sum(np.abs(theta))
        else:
            reg = 0.0

        costs.append(mse + reg)

    return np.array(costs)
