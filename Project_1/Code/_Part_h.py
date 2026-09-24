from sgd_methods import *
from general_functions import *
from gradient_descent_methods import *
from lasso_methods import * 

import matplotlib.pyplot as plt

degree = 5
lam = 1e-2
n = 100

#making data
x_raw, y_raw = make_data(n = n)
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, y = scaling(X_raw, y_raw, split_data = False)
theta0 = np.zeros(X.shape[1])

theta_cf = closed_form(X, y, lmbda = lam)
H_OLS = 2.0 / len(y) * X.T @ X
H_Ridge = 2.0 / len(y) * X.T @ X + 2 * lam * np.eye(X.shape[1])

gamma_max_OLS = 2.0 / np.linalg.eigvalsh(H_OLS).max()
gamma_max_Ridge = 2.0 / np.linalg.eigvalsh(H_Ridge).max()

gamma_list = np.linspace(0.001, 1.2 * max(gamma_max_OLS, gamma_max_Ridge), 40)

distances = []
for gamma in gamma_list:
    hist = sgd(X, y, n_epochs=100, batch_size=5, gamma=gamma)
    d = np.linalg.norm(hist - theta_cf, axis=1)
    distances.append(d)

final_distances = [d[-1] for d in distances]

plt.plot(gamma_list, final_distances, "o-", label = "sgd - cf")
plt.axvline(gamma_max_OLS, color = 'C0', linestyle = "--", alpha = 0.7, label = f'OLS limit $\\gamma$ = {gamma_max_OLS:.3f}')
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\|\boldsymbol{\theta}_{sgd}-\hat{\boldsymbol{\theta}}_{cf}\|_2$")
#plt.yscale("log")
#plt.xscale("log")
plt.grid()
plt.legend()
plt.show()


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

"""Gradient descent"""
history_lasso_gd, n_lasso_gd = lasso_gd(X, y, gamma_max_Ridge, lmbda = lam)

"""Coordinate descent"""
history_lasso_cd, n_lasso_cd = lasso_coordinate_descent(X, y, lmbda=lam)

"""Cost functions of OLS, Ridge, Lasso for gd and sgd"""
cost_OLS = cost_history(history_OLS, X, y, penalty="None")
cost_Ridge = cost_history(history_Ridge, X, y, lmbda=lam, penalty="L2")
cost_Lasso_gd = cost_history(history_lasso_gd, X, y, lmbda=lam, penalty="L1")
cost_Lasso_cd = cost_history(history_lasso_cd, X, y, lmbda=lam, penalty="L1")


history_OLS_sgd = sgd(X, y, gamma = gamma_max_OLS)
history_Ridge_sgd = sgd(X, y, gamma = gamma_max_Ridge, lmbda = lam)

cost_OLS_sgd = cost_history(history_OLS_sgd, X, y, penalty = "None")
cost_Ridge_sgd = cost_history(history_Ridge_sgd, X, y, lmbda = lam, penalty = "L2")

"""Plot of cost vs iteration"""
plt.plot(cost_Lasso_gd, label = "Lasso gradient descent")
plt.plot(cost_Lasso_cd, label = "Lasso coordinate descent")
plt.plot(cost_OLS, label = "OLS")
plt.plot(cost_Ridge, label = "Ridge")
#plt.plot(cost_OLS_sgd, label = "OLS sgd")
#plt.plot(cost_Ridge_sgd, label = "Ridge sgd")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of gradient descent")
plt.xscale("log")
#plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

plt.plot(cost_OLS_sgd, label = "OLS sgd")
plt.plot(cost_Ridge_sgd, label = "Ridge sgd")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of stochastic gradient descent")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()