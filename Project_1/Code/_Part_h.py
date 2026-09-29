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


"""Studying varying learning rate"""
gamma_max_OLS = 2.0 / np.linalg.eigvalsh(H_OLS).max()
gammas_OLS = [0.01 * gamma_max_OLS, 0.1 * gamma_max_OLS, 0.5 * gamma_max_OLS]
labels_gamma = [r"$0.01\, \gamma_{\max}$", r"$0.1\, \gamma_{\max}$", r"$0.5\, \gamma_{\max}$"]

for gamma, label in zip(gammas_OLS, labels_gamma):
    hist, n_gamma = sgd(X, y, n_epochs=100, batch_size=5, gamma=gamma)
    cost_gamma = cost_history(hist, X, y, penalty = "None")

    plt.plot(cost_gamma, label = label)

"""Plot of cost as for different learning rates"""
plt.title("Cost function for different learning rates")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


"""Studying varying batch sizes"""
batches = [10, 20, 30, 40]
labels_M = [f"$M = {batches[0]}$", f"$M = {batches[1]}$", f"$M = {batches[2]}$", f"$M = {batches[3]}$"]

for M, label in zip(batches, labels_M):
    hist, n_M = sgd(X, y, n_epochs=100, batch_size=M, gamma=gamma_max_OLS)
    cost_M = cost_history(hist, X, y, penalty = "None")
    
    plt.plot(cost_M, label = label)
    
"""Plot of cost as for different batch sizes"""
plt.title("Cost function for different batch sizes")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

"""Studying varying epoch number"""
epochs = [1, 10, 100]
labels_n = [f"$n = {epochs[0]}$", f"$n = {epochs[1]}$", f"$n = {epochs[2]}$"]

for n_ep, label in zip(epochs, labels_n):
    hist, n_s = sgd(X, y, n_epochs=n_ep, batch_size=5, gamma=gamma_max_OLS)
    cost_n = cost_history(hist, X, y, penalty = "None")
        
    plt.plot(cost_n, label = label)
 
"""Plot of cost as for different number of epochs"""
plt.title("Cost function for different number of epochs")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show() 


"""Gradient descent for OLS and Ridge"""
#the largest safe learning rate for plain gradient descent: gamma < 2 / lambda_max(hessian)
H_OLS = 2.0 / len(y) * X.T @ X
H_Ridge = 2.0 / len(y) * X.T @ X + 2 * lam * np.eye(X.shape[1])

gamma_max_OLS = 2.0 / np.linalg.eigvalsh(H_OLS).max()
gamma_max_Ridge = 2.0 / np.linalg.eigvalsh(H_Ridge).max()

history_OLS, n_OLS = gradient_descent(X, y, gamma = gamma_max_OLS)
history_Ridge, n_Ridge = gradient_descent(X, y, lmbda = lam, gamma = gamma_max_Ridge)
cost_OLS = cost_history(history_OLS, X, y, penalty="None")
cost_Ridge = cost_history(history_Ridge, X, y, lmbda=lam, penalty="L2")

history_OLS_sgd, n_OLS_sgd = sgd(X, y )
history_Ridge_sgd, n_Ridge_sgd = sgd(X, y, lmbda = lam)
cost_OLS_sgd = cost_history(history_OLS_sgd, X, y, penalty = "None")
cost_Ridge_sgd = cost_history(history_Ridge_sgd, X, y, lmbda = lam, penalty = "L2")


print("OLS: GD vs. SGD------------------------------")
print(f'iterations to converge: GS: {n_OLS} SGD: {n_OLS_sgd}')
print(f'DG: theta = {history_OLS[-1]}')
print(f'SDG: theta = {history_OLS_sgd[-1]}')
print(f"theta: |sgd - gd| = {np.max(np.abs(history_OLS[-1] - history_OLS_sgd[-1])):.2e} ")


print("Ridge: GD vs. SGD------------------------------")
print(f'iterations to converge: GS: {n_Ridge} SGD: {n_Ridge_sgd}')
print(f'DG: theta = {history_Ridge[-1]}')
print(f'SDG: theta = {history_Ridge_sgd[-1]}')
print(f"theta: |sgd - gd| = {np.max(np.abs(history_Ridge[-1] - history_Ridge_sgd[-1])):.2e} ")


"""Plot of cost vs iteration"""
plt.plot(cost_OLS, label = "OLS GD", color = "C0")
plt.plot(cost_OLS_sgd, label = "OLS SGD", color = "C0", linestyle = "--")
plt.plot(cost_Ridge, label = "Ridge GD", color = "C1")
plt.plot(cost_Ridge_sgd, label = "Ridge SGD", color = "C1", linestyle = "--")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of stochastic gradient descent")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()