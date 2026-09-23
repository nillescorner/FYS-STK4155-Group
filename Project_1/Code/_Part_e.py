from general_functions import *
from regression_methods import *
from gradient_descent_methods import *

from jax import grad

n = 100
sigma = 0.1
degree = 5
lam = 1e-2

#making data
x_raw, y_raw = make_data(n = n, noise = sigma)
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, y = scaling(X_raw, y_raw, split_data = False)
theta0 = np.zeros(X.shape[1])


"""Analytical for OLS and ridge"""
gradient_OLS = gradient(theta0, X, y)
gradient_Ridge = gradient(theta0, X, y, lam= lam)

"""Automatic differentiation"""
gradient_jax = grad(cost)

gradient_OLS_jax = gradient_jax(theta0, X, y)
gradient_Ridge_jax = gradient_jax(theta0, X, y, lmbda = lam)

"""Difference between analytical and automatic differentiation"""
print(f'OLS:    |AD - analytical| = {np.max(np.abs(gradient_OLS_jax - gradient_OLS)):.2e}')
print(f'Ridge:  |AD - analytical| = {np.max(np.abs(gradient_Ridge_jax - gradient_Ridge)):.2e}')


"""Gradient descent for OLS and Ridge"""
#the largest safe learning rate for plain gradient descent: gamma < 2 / lambda_max(hessian)
H_OLS = 2.0 / len(y) * X.T @ X
H_Ridge = 2.0 / len(y) * X.T @ X + 2 * lam * np.eye(X.shape[1])

gamma_max_OLS = 2.0 / np.linalg.eigvalsh(H_OLS).max()
gamma_max_Ridge = 2.0 / np.linalg.eigvalsh(H_Ridge).max()

theta_OLS, n_OLS = gradient_descent(X, y, gamma_max_OLS)
theta_Ridge, n_Ridge = gradient_descent(X, y, gamma_max_Ridge, lam = lam)

