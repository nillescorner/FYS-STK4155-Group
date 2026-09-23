from general_functions import *
from regression_methods import *
from gradient_descent_methods import *

from jax import grad
import matplotlib.pyplot as plt

degree = 5
lam = 1e-2

#making data
x_raw, y_raw = make_data()
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, y = scaling(X_raw, y_raw, split_data = False)
theta0 = np.zeros(X.shape[1])

"""Analytical for OLS and ridge"""
gradient_OLS = gradient(theta0, X, y)
gradient_Ridge = gradient(theta0, X, y, lam)

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

history_OLS, n_OLS = gradient_descent(X, y, gamma_max_OLS)
history_Ridge, n_Ridge = gradient_descent(X, y, gamma_max_Ridge, lam)
theta_OLS = history_OLS[-1]
theta_Ridge = history_Ridge[-1]

"""Cost of OLS and Ridge, for each theta value"""
cost_OLS = np.array([cost(theta_i, X, y) for theta_i in history_OLS])
cost_Ridge = np.array([cost(theta_i, X, y, lmbda= lam) for theta_i in history_Ridge])

"""Plot of cost vs iteration"""
plt.plot(cost_OLS, label = "OLS")
plt.plot(cost_Ridge, label = "Ridge")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of gradient descent")
plt.xscale("log")
#plt.yscale("log")
plt.grid()
plt.legend()
plt.show()


"""Closed form to compare to"""
theta_cf_OLS = closed_form(X, y)
theta_cf_Ridge = closed_form(X, y, lmbda = lam)

print(f'OLS:    |analytical - closed form| = {np.max(np.abs(theta_OLS - theta_cf_OLS)):.3e}, converges after {n_OLS} iterations')
print(f'Ridge:  |analytical - closed form| = {np.max(np.abs(theta_Ridge - theta_cf_Ridge)):.3e}, converges after {n_Ridge} iterations')

dist_OLS = np.linalg.norm(history_OLS - theta_cf_OLS, axis = 1)
dist_Ridge = np.linalg.norm(history_Ridge - theta_cf_Ridge, axis = 1)

"""Plot of difference between analytical and closed form"""
plt.plot(dist_OLS, label = "OLS")
plt.plot(dist_Ridge, label = "Ridge")
plt.xlabel("Iteration")
plt.ylabel(r"$\|\boldsymbol{\theta}_{gd}-\hat{\boldsymbol{\theta}}_{cf}\|_2$")
plt.title("Convergense of gradient descent against closed form")
plt.legend()
#plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.show()

"""Plot of convergence as a func of different gammas"""
"""OLS"""
gammas = [0.1 * gamma_max_OLS, 0.5 * gamma_max_OLS, 0.99 * gamma_max_OLS, 1.001 * gamma_max_OLS]
labels = [r"$0.1\, \gamma_{\max}$", r"$0.5\, \gamma_{\max}$", r"$0.99\, \gamma_{\max}$", r"$1.001\, \gamma_{\max}$"]

for gamma, label in zip(gammas, labels):
    hist, n_ = gradient_descent(X, y, gamma)
    dist_gam_OLS = np.linalg.norm(hist - theta_cf_OLS, axis = 1)

    plt.plot(dist_gam_OLS, label = label, ls = "--" if gamma > gamma_max_OLS else "-")

plt.ylabel(r"$\|\boldsymbol{\theta}_{gd}-\hat{\boldsymbol{\theta}}_{cf}\|_2$")
plt.title("Convergense of gradient descent: OLS")
plt.ylim(1e-8, 1e4)
plt.legend()
plt.yscale("log")
#plt.xscale("log")
plt.grid()
plt.show()

"""Ridge"""
for gamma, label in zip(gammas, labels):
    hist, n_ = gradient_descent(X, y, gamma, lam)
    dist_gam_Ridge = np.linalg.norm(hist - theta_cf_Ridge, axis = 1)

    plt.plot(dist_gam_Ridge, label = label, ls = "--" if gamma > gamma_max_OLS else "-")

plt.ylabel(r"$\|\boldsymbol{\theta}_{gd}-\hat{\boldsymbol{\theta}}_{cf}\|_2$")
plt.title("Convergense of gradient descent: Ridge")
plt.ylim(1e-8, 1e4)
plt.legend()
plt.yscale("log")
#plt.xscale("log")
plt.grid()
plt.show()

