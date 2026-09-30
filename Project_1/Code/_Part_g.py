from lasso_methods import *
from general_functions import *
from gradient_descent_methods import *

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


degree = 5
lam = 1e-2
n = 100

#making data
x_raw, y_raw = make_data(n = n)
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, y = scaling(X_raw, y_raw, split_data = False)
theta0 = np.zeros(X.shape[1])


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
cost_OLS = cost_history(history_OLS, X, y, penalty="None")
cost_Ridge = cost_history(history_Ridge, X, y, lmbda=lam, penalty="L2")


"""Gradient descent"""
history_lasso_gd, n_lasso_gd = lasso_gd(X, y, gamma_max_Ridge, lmbda = lam)

"""Coordinate descent"""
history_lasso_cd, n_lasso_cd = lasso_coordinate_descent(X, y, lmbda=lam)

cost_Lasso_gd = cost_history(history_lasso_gd, X, y, lmbda=lam, penalty="L1")
cost_Lasso_cd = cost_history(history_lasso_cd, X, y, lmbda=lam, penalty="L1")

"""SciKit-learn lasso"""
model = Lasso(alpha=lam / 2.0, fit_intercept=False, max_iter=40000, tol=1e-12).fit(X, y)
theta_skl_lasso = model.coef_
n_iter_skl = model.n_iter_

print("Lasso convergence")
print("SciKit-learn------------------------------------")
print(f'iterations to converge: {n_iter_skl}')
print(f'theta_lasso = {theta_skl_lasso}')

print("Gradient descent--------------------------------")
print(f'iterations to converge: {n_lasso_gd}')
print(f'theta_lasso = {history_lasso_gd[-1]}')
print(f"theta: |skl - gd| = {np.max(np.abs(theta_skl_lasso - history_lasso_gd[-1])):.2e} ")

print("Coordinate descent------------------------------")
print(f'iterations to converge: {n_lasso_cd}')
print(f'theta_lasso = {history_lasso_cd[-1]}')
print(f"theta: |skl - cd| = {np.max(np.abs(theta_skl_lasso - history_lasso_cd[-1])):.2e} ")


"""Plot of cost vs iteration"""
plt.plot(cost_Lasso_gd, label = "Lasso gradient descent", color = "#7AD2EF")
plt.plot(cost_Lasso_cd, label = "Lasso coordinate descent", color = "#89EF82")
plt.plot(cost_OLS, label = "OLS", color = "#F433DA")
plt.plot(cost_Ridge, label = "Ridge", color = "#7326E6")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of gradient descent")
plt.xscale("log")
#plt.yscale("log")
plt.grid()
plt.legend()
#plt.savefig("Part_g_convergence_Lasso.png")
plt.show()