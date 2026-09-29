
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
X, y = scaling(X_raw, y_raw, split_data = False)
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
plt.plot(dist_OLS, label = "OLS")
plt.plot(dist_Ridge, label = "Ridge")
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

plt.plot(gamma_list, diffs_ols, label='OLS')
plt.plot(gamma_list, diffs_ridge, label='Ridge')
plt.axvline(gamma_max_OLS, color = 'C0', linestyle = "--", alpha = 0.7, label = f'OLS limit $\\gamma$ = {gamma_max_OLS:.3f}')
plt.axvline(gamma_max_Ridge, color = 'C1', linestyle = "--", alpha = 0.7, label = f'Ridge limit $\\gamma$ = {gamma_max_Ridge:.3f}')
plt.xlabel('Learning rate (γ)')
plt.ylabel(r'$\|\theta_{gd} - \theta_{cf}\|$')
plt.yscale('log')  
plt.title('Final parameter error vs. learning rate')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
#plt.savefig("Part_e_error_learningrate.png")
plt.show()


plt.plot(gamma_list, n_ols, label='OLS')
plt.plot(gamma_list, n_ridge, label='Ridge')
plt.axvline(gamma_max_OLS, color = 'C0', linestyle = "--", alpha = 0.7, label = f'OLS limit $\\gamma$ = {gamma_max_OLS:.3f}')
plt.axvline(gamma_max_Ridge, color = 'C1', linestyle = "--", alpha = 0.7, label = f'Ridge limit $\\gamma$ = {gamma_max_Ridge:.3f}')
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

for gamma, label in zip(gammas_OLS, labels):
    hist, n_ = gradient_descent(X, y, gamma)
    dist_gam_OLS = np.linalg.norm(hist - theta_cf_OLS, axis = 1)

    plt.plot(dist_gam_OLS, label = label, ls = "--" if gamma > gamma_max_OLS else "-")

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

for gamma, label in zip(gammas_Ridge, labels):
    hist, n_ = gradient_descent(X, y, gamma, lam)
    dist_gam_Ridge = np.linalg.norm(hist - theta_cf_Ridge, axis = 1)

    plt.plot(dist_gam_Ridge, label = label, ls = "--" if gamma > gamma_max_Ridge else "-")

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
