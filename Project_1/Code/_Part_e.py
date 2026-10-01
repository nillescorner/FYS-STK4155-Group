
"""
The file conatins the code used to derive the plots of the cost functions for OLS and Ridge regression,
and has been compared to the closed form solutions of the different methods. The difference between 
the gradient descent method and closed form solution has been plotted as a function of the learning rate.
The convergence of the different methods for different learning rates is also plotted.

The gradients were found using the gradient function, and the gradient descent found from the gradient_descent
function from gradient_descent_methods.py. The closed form solution is found using the closed_form function
also from gradient_descent_methods.py.

"""
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
X, X_test, y, y_test = scaling(X_raw, y_raw, split_data = True)
theta0 = np.zeros(X.shape[1])

"""Analytical for OLS and ridge"""
gradient_OLS = gradient(theta0, X, y)
gradient_Ridge = gradient(theta0, X, y, lam)


"""Automatic differentiation"""
gradient_jax = grad(cost)

gradient_OLS_jax = gradient_jax(theta0, X, y, penalty = "None")
gradient_Ridge_jax = gradient_jax(theta0, X, y, lmbda = lam, penalty = "L2")

"""Difference between analytical and automatic differentiation"""
print(f'OLS:    |AD - analytical| = {np.max(np.abs(gradient_OLS_jax - gradient_OLS)):.4e}')
print(f'Ridge:  |AD - analytical| = {np.max(np.abs(gradient_Ridge_jax - gradient_Ridge)):.4e}')


"""Gradient descent for OLS and Ridge"""
#the largest safe learning rate for plain gradient descent: gamma < 2 / lambda_max(hessian)
H_OLS = hessian_eigs(X)
H_Ridge = hessian_eigs(X, lmbda= lam)

gamma_max_OLS = 2.0 / H_OLS.max()
gamma_max_Ridge = 2.0 / H_Ridge.max()

history_OLS, n_OLS = gradient_descent(X, y, gamma_max_OLS)
history_Ridge, n_Ridge = gradient_descent(X, y, gamma_max_Ridge, lam)
theta_OLS = history_OLS[-1]
theta_Ridge = history_Ridge[-1]

"""Cost of OLS and Ridge, for each theta value"""
cost_OLS = cost_history(history_OLS, X, y, penalty="None")
cost_Ridge = cost_history(history_Ridge, X, y, lmbda=lam, penalty="L2")

"""Plot of cost vs iteration"""
plt.plot(cost_OLS, label = "OLS", color = "#F433DA")
plt.plot(cost_Ridge, label = "Ridge", color = "#7326E6")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of gradient descent")
plt.xscale("log")
#plt.yscale("log")
plt.grid()
plt.legend()
#plt.savefig("Part_e_costfunc_OLS_Ridge.png")
plt.show()

"""Closed form to compare to"""
theta_cf_OLS = closed_form(X, y)
theta_cf_Ridge = closed_form(X, y, lmbda = lam)

print(f'OLS:    |analytical - closed form| = {np.max(np.abs(theta_OLS - theta_cf_OLS)):.3e}, converges after {n_OLS} iterations')
print(f'Ridge:  |analytical - closed form| = {np.max(np.abs(theta_Ridge - theta_cf_Ridge)):.3e}, converges after {n_Ridge} iterations')

dist_OLS = np.linalg.norm(history_OLS - theta_cf_OLS, axis = 1)
dist_Ridge = np.linalg.norm(history_Ridge - theta_cf_Ridge, axis = 1)

"""Plot of difference between analytical and closed form"""
plt.plot(dist_OLS, label = "OLS", color = "#F433DA")
plt.plot(dist_Ridge, label = "Ridge", color = "#7326E6")
plt.xlabel("Iteration")
plt.ylabel(r"$\|\boldsymbol{\theta}_{gd}-\hat{\boldsymbol{\theta}}_{cf}\|_2$")
plt.title("Convergense of gradient descent against closed form")
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.grid()
#plt.savefig("Part_e_convergence_gd_cf.png")
plt.show()


"""Study of different learning rates"""
gamma_list = np.linspace(0.001, 1.2 * max(gamma_max_OLS, gamma_max_Ridge), 40)

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

plt.plot(gamma_list, diffs_ols, label='OLS', color = "#F433DA")
plt.plot(gamma_list, diffs_ridge, label='Ridge', color = "#7326E6")
plt.axvline(gamma_max_OLS, color = "#F433DA", linestyle = "--", alpha = 0.5, label = f'OLS limit $\\gamma$ = {gamma_max_OLS:.3f}')
plt.axvline(gamma_max_Ridge, color = "#7326E6", linestyle = "--", alpha = 0.5, label = f'Ridge limit $\\gamma$ = {gamma_max_Ridge:.3f}')
plt.xlabel('Learning rate (γ)')
plt.ylabel(r'$\|\theta_{gd} - \theta_{cf}\|$')
plt.yscale('log')  
plt.title('Final parameter error vs. learning rate')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
#plt.savefig("Part_e_error_learningrate.png")
plt.show()


plt.plot(gamma_list, n_ols, label='OLS', color = "#F433DA")
plt.plot(gamma_list, n_ridge, label='Ridge', color = "#7326E6")
plt.axvline(gamma_max_OLS, color = "#F433DA", linestyle = "--", alpha = 0.5, label = f'OLS limit $\\gamma$ = {gamma_max_OLS:.3f}')
plt.axvline(gamma_max_Ridge, color = "#7326E6", linestyle = "--", alpha = 0.5, label = f'Ridge limit $\\gamma$ = {gamma_max_Ridge:.3f}')
plt.axvline(best_gamma_OLS, color = 'grey', linestyle = "--", alpha = 0.7, label = f'OLS min iteration $\\gamma$ = {best_gamma_OLS:.3f}')
plt.axvline(best_gamma_Ridge, color = 'grey', linestyle = "--", alpha = 0.7, label = f'Ridge min iterations $\\gamma$ = {best_gamma_Ridge:.3f}')
plt.xlabel('Learning rate (γ)')
plt.ylabel('Iterations')
plt.yscale('log')  
plt.title('Iterations vs. learning rate')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
#plt.savefig("Part_e_iteration_learningrate.png")
plt.show()


"""Plot of convergence as a func of different learning rate"""
"""OLS"""
gammas_OLS = [0.1 * gamma_max_OLS, 0.5 * gamma_max_OLS, 1.001 * gamma_max_OLS, best_gamma_OLS]
labels = [r"$0.1\, \gamma_{\max}$", r"$0.5\, \gamma_{\max}$", r"$1.001\, \gamma_{\max}$", r"$\gamma_{best}$"]
colors = ["#F433DA", "#7326E6", "#26B3E6", "#E62663"]

for gamma, label, color in zip(gammas_OLS, labels, colors):
    hist, n_ = gradient_descent(X, y, gamma)
    dist_gam_OLS = np.linalg.norm(hist - theta_cf_OLS, axis = 1)

    plt.plot(dist_gam_OLS, label = label, ls = "--" if gamma > gamma_max_OLS else "-", color = color)

plt.ylabel(r"$\|\boldsymbol{\theta}_{gd}-\hat{\boldsymbol{\theta}}_{cf}\|_2$")
plt.xlabel("Iterations")
plt.title("Convergense of gradient descent: OLS")
plt.ylim(1e-8, 1e4)
plt.legend()
plt.yscale("log")
#plt.xscale("log")
plt.grid()
#plt.savefig("Part_e_convergence_learningrate_OLS.png")
plt.show()

"""Ridge"""
gammas_Ridge = [0.1 * gamma_max_Ridge, 0.5 * gamma_max_Ridge, 1.001 * gamma_max_Ridge, best_gamma_Ridge]

for gamma, label, color in zip(gammas_Ridge, labels, colors):
    hist, n_ = gradient_descent(X, y, gamma, lam)
    dist_gam_Ridge = np.linalg.norm(hist - theta_cf_Ridge, axis = 1)

    plt.plot(dist_gam_Ridge, label = label, ls = "--" if gamma > gamma_max_Ridge else "-", color = color)

plt.ylabel(r"$\|\boldsymbol{\theta}_{gd}-\hat{\boldsymbol{\theta}}_{cf}\|_2$")
plt.xlabel("Iterations")
plt.title("Convergense of gradient descent: Ridge")
plt.ylim(1e-8, 1e4)
plt.legend()
plt.yscale("log")
#plt.xscale("log")
plt.grid()
#plt.savefig("Part_e_convergence_learningrate_Ridge.png")
plt.show()


"""Comparing the trained data and the test data"""
n_test = X_test.shape[0]
cost_test_OLS = cost(theta_OLS, X_test, y_test, penalty = "None")
cost_test_Ridge = cost(theta_Ridge, X_test, y_test, lmbda=lam, penalty = "L2")
test_vals = [cost_test_OLS, cost_test_Ridge]

n_train = X.shape[0]
cost_train_OLS = cost(theta_OLS, X, y, penalty = "None")
cost_train_Ridge = cost(theta_Ridge, X, y, lmbda=lam, penalty = "L2")
train_vals = [cost_train_OLS, cost_train_Ridge]

labels = ["OLS", "Ridge"]
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, train_vals, width, label="Train MSE", color = "#F433DA")
plt.bar(x + width/2, test_vals, width, label="Test MSE", color = "#7326E6")
plt.xticks(x, labels)
plt.ylabel("MSE")
plt.title("Train vs. test MSE: OLS vs Ridge")
plt.legend()
plt.grid(axis='y', alpha=0.3)
#plt.savefig("Part_e_hist_testtrain.png")
plt.show()

print("Test-set evaluation-------------------------------")
print(f"OLS:   train={cost_train_OLS:.4e}, test={cost_test_OLS:.4e}, gap={cost_test_OLS - cost_train_OLS:.4e}")
print(f"Ridge: train={cost_train_Ridge:.4e}, test={cost_test_Ridge:.4e}, gap={cost_test_Ridge - cost_train_Ridge:.4e}")


"""For different lambda values, comparing test and train"""
lambdas = np.logspace(-4, 2, 30)
train_errs, test_errs = [], []

for lm in lambdas:
    H_Ridge_lm = hessian_eigs(X, lmbda= lm)
    gamma_max_Ridge_lm = 2.0 / H_Ridge_lm.max()

    history_Ridge, n_Ridge = gradient_descent(X, y, gamma_max_Ridge_lm, lm)
    theta_lm = history_Ridge[-1]
    train_errs.append(cost(theta_lm, X, y, lmbda=lm))
    test_errs.append(cost(theta_lm, X_test, y_test, lmbda=lm))

plt.plot(lambdas, train_errs, label="Train MSE", color = "#F433DA")
plt.plot(lambdas, test_errs, label="Test MSE", color = "#7326E6")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\lambda$")
plt.ylabel("Cost")
plt.title("Train vs test error across regularization strength")
plt.legend()
plt.tight_layout()
plt.grid()
#plt.savefig("Part_e_cost_lambda_testtrain.png")
plt.show()

