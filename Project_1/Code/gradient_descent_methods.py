"""Cost functions, gradients (analytical + autodiff) and plain gradient descent"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # 64-bit floats, as in numpy


def cost_ols(theta, X, y):
    return jnp.mean((y - X @ theta)**2)

def cost_ridge(theta, X, y, lmbda):
    return jnp.mean((y - X @ theta)**2) + lmbda * jnp.sum(theta**2)


# Q from Hannah: No reason to define different cost  functions no? we can just set lmbda=0
#NOTE: i put this here so you chan change things later
def cost(theta, X, y, lmbda=0):
    """OLS cost when lmbda=0, otherwise Ridge cost"""
    return jnp.mean((y - X @ theta)**2) + lmbda * jnp.sum(theta**2)


"""Analytical"""
def gradient(theta, X, y, lam=0.0):
    """Eqs. (4.13) and (4.17): the gradient of (1/n)||X theta - y||^2 + lambda theta^T theta."""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta


"""Gradient descent"""
def gradient_descent(X, y, gamma, lam=0.0, num_iters=10000, tol=1e-8, theta0=None):
    """Plain gradient descent, Eq. (4.15). Returns the iterates and the number of steps."""
    p = X.shape[1]
    theta = np.zeros(p) if theta0 is None else theta0.copy()
    history = [theta.copy()]
    for t in range(num_iters):
        g = gradient(theta, X, y, lam)
        theta = theta - gamma * g
        history.append(theta.copy())

        if np.linalg.norm(g) < tol:
            break
    return np.array(history), t + 1
