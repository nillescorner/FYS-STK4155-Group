from lasso_methods import *
from general_functions import *
from gradient_descent_methods import *

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from plots import plot_theta

degree = 5
lam = 1e-2
n = 1000

#making data
x_raw, y_raw = make_data(n = n)
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, X_test, y, y_test = scaling(X_raw, y_raw, split_data = True)
theta0 = np.zeros(X.shape[1])


"""Gradient descent for OLS and Ridge"""
theta_cf_OLS = closed_form(X, y)
theta_cf_Ridge = closed_form(X, y, lmbda = lam)
gamma_list = np.linspace(0.001, 0.4, 40)

diffs_ols, diffs_ridge = [], []
n_ols, n_ridge = [], []

for gam in gamma_list:
    hist_ols, n_steps_ols = gradient_descent(X, y, gam)
    hist_Ridge, n_steps_Ridge = gradient_descent(X, y, gam, lam)

    diffs_ols.append(np.linalg.norm(hist_ols[-1] - theta_cf_OLS))
    diffs_ridge.append(np.linalg.norm(hist_Ridge[-1] - theta_cf_Ridge))
    n_ols.append(n_steps_ols)
    n_ridge.append(n_steps_Ridge)

n_min_OLS = np.argmin(n_ols)
n_min_Ridge = np.argmin(n_ridge)

best_gamma_OLS = gamma_list[n_min_OLS]
best_gamma_Ridge = gamma_list[n_min_Ridge]

history_OLS, n_OLS = gradient_descent(X, y, best_gamma_OLS)
history_Ridge, n_Ridge = gradient_descent(X, y, best_gamma_Ridge, lam)
theta_OLS = history_OLS[-1]
theta_Ridge = history_Ridge[-1]
cost_OLS = cost_history(history_OLS, X, y, penalty="None")
cost_Ridge = cost_history(history_Ridge, X, y, lmbda=lam, penalty="L2")


"""Gradient descent"""
history_lasso_gd, n_lasso_gd = lasso_gd(X, y, best_gamma_Ridge, lmbda = lam)

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
plt.plot(cost_Lasso_gd, label = "Lasso gradient descent", color = "#5DA0B6")
plt.plot(cost_Lasso_cd, label = "Lasso coordinate descent", color = "#65AF60")
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


"""Comparing train and test MSE across OLS, Ridge, and Lasso"""

n_train = X.shape[0]
n_test = X_test.shape[0]

theta_lasso_gd = history_lasso_gd[-1]

methods = {
    "OLS": theta_OLS,
    "Ridge": theta_Ridge,
    "Lasso (GD)": theta_lasso_gd,}

print("Test vs train MSE for different regression methods--------------------")
train_vals, test_vals = [], []
for name, theta in methods.items():
    tr = cost(theta, X, y)
    te = cost(theta, X_test, y_test)
    train_vals.append(tr)
    test_vals.append(te)
    print(f"{name:16s}: train MSE = {tr:.4e}, test MSE = {te:.4e}, gap = {te - tr:.4e}")

labels = list(methods.keys())
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, train_vals, width, label="Train MSE", color="#F433DA", alpha = 0.7)
plt.bar(x + width/2, test_vals, width, label="Test MSE", color="#7326E6", alpha = 0.7)
plt.xticks(x, labels, rotation=15)
plt.ylabel("MSE")
plt.title("Train vs. test MSE: OLS, Ridge, Lasso")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
#plt.savefig("Part_g_hist_testtrain_lasso.png")
plt.show()

"""For different lambda values, comparing test and train"""
lambdas = np.logspace(-4, 2, 30)
train_errs_Ridge, test_errs_Ridge = [], []
train_errs_Lasso, test_errs_Lasso = [], []
thetas_Ridge, thetas_Lasso = [], []          


for lm in lambdas:
    H_Ridge_lm = hessian_eigs(X, lmbda= lm)
    gamma_max_Ridge_lm = 2.0 / H_Ridge_lm.max()

    history_Ridge, n_Ridge = gradient_descent(X, y, gamma_max_Ridge_lm, lm)
    history_lasso_gd, n_lasso_gd = lasso_gd(X, y, gamma_max_Ridge_lm, lmbda = lm)
    theta_lm_Ridge = history_Ridge[-1]
    theta_lm_Lasso = history_lasso_gd[-1]

    train_errs_Ridge.append(cost(theta_lm_Ridge, X, y, lmbda=lm))
    test_errs_Ridge.append(cost(theta_lm_Ridge, X_test, y_test, lmbda=lm))

    train_errs_Lasso.append(cost(theta_lm_Lasso, X, y, lmbda=lm, penalty = "L1"))
    test_errs_Lasso.append(cost(theta_lm_Lasso, X_test, y_test, lmbda=lm, penalty = "L1"))

    thetas_Ridge.append(theta_lm_Ridge)     
    thetas_Lasso.append(theta_lm_Lasso)     

plt.plot(lambdas, train_errs_Ridge, label="Train Ridge", color = "#F433DA")
plt.plot(lambdas, test_errs_Ridge, label="Test Ridge", color = "#F433DA", linestyle = "--")
plt.plot(lambdas, train_errs_Lasso, label="Train Lasso", color = "#7326E6")
plt.plot(lambdas, test_errs_Lasso, label="Test Lasso", color = "#7326E6", linestyle = "--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\lambda$")
plt.ylabel("Cost")
plt.title("Train vs test error across regularization strength")
plt.legend()
plt.tight_layout()
plt.grid()
#plt.savefig("Part_g_cost_lambda_testtrain.png")
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

plot_theta(thetas_Ridge, lambdas, xlabel=r'$\lambda$', title='Ridge coefficients', intercept=False, xlog=True, ax=ax1)
plot_theta(thetas_Lasso, lambdas, xlabel=r'$\lambda$', title='Lasso coefficients', intercept=False, xlog=True, ax=ax2)
#plt.savefig("Part_g_theta_lambda.png")
#plt.savefig(fname = "figs/Part_g_CostVsIterationLasso_GradientAndCoordinateDescent.pdf") what is this??
plt.show()