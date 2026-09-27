"""Cost functions, gradients (analytical + autodiff) and plain gradient descent"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # 64-bit floats, as in numpy

def closed_form(X, y, lmbda=0.0):
    """Eq. (3.44) with the 1/n convention of Eq. (3.95): (X^T X + n lambda I)^-1 X^T y."""
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lmbda * np.eye(p), X.T @ y)

def cost(theta, X, y, lmbda=0.0, penalty = "L2"):
    """OLS cost when lmbda=0, otherwise Ridge cost"""
    n, p = X.shape

    mse = (1.0 / n) * np.sum((X @ theta - y) **2)

    #penalty for ridge
    if penalty == "L2": 
        reg = lmbda * np.sum(theta ** 2)

    #penalty for lasso
    elif penalty == "L1":
        reg = lmbda * np.sum(np.abs(theta))

    #OLS
    else: 
        reg = 0.0

    return mse + reg

def cost_history(history, X, y, lmbda=0.0, penalty="L2"):
    """Cost at every theta in a gradient descent history. Returns an array."""
    return np.array([cost(theta, X, y, lmbda, penalty) for theta in history])


def gradient(theta, X, y, lmbda=0.0):
    """Eqs. (4.13) and (4.17): the gradient of (1/n)||X theta - y||^2 + lambda theta^T theta."""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lmbda * theta

def gradient_descent(X, y, gamma, lmbda=0.0, num_iters=10000, tol=1e-8, theta0=None):
    """Plain gradient descent, Eq. (4.15). Returns the iterates and the number of steps."""
    p = X.shape[1]
    theta = np.zeros(p) if theta0 is None else theta0.copy()
    history = [theta.copy()]
    for t in range(num_iters):
        g = gradient(theta, X, y, lmbda)
        theta = theta - gamma * g
        history.append(theta.copy())

        if np.linalg.norm(g) < tol:
            break
    return np.array(history), t + 1
