from sgd_methods import *
from general_functions import *
from gradient_descent_methods import *
from lasso_methods import * 

import matplotlib.pyplot as plt

degree = 5
lam = 1e-2
n = 1000

#making data
x_raw, y_raw = make_data(n = n)
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, X_test, y, y_test = scaling(X_raw, y_raw, split_data = True)
theta0 = np.zeros(X.shape[1])

theta_cf = closed_form(X, y, lmbda = lam)
H_OLS = 2.0 / len(y) * X.T @ X
H_Ridge = 2.0 / len(y) * X.T @ X + 2 * lam * np.eye(X.shape[1])


"""Studying varying learning rate"""
gamma_max_OLS = 2.0 / np.linalg.eigvalsh(H_OLS).max()
gammas_OLS = [0.01 * gamma_max_OLS, 0.1 * gamma_max_OLS, 0.5 * gamma_max_OLS]
labels_gamma = [r"$0.01\, \gamma_{\max}$", r"$0.1\, \gamma_{\max}$", r"$0.5\, \gamma_{\max}$"]
colors = ["#F433DA", "#7326E6", "#84BCED"]

for gamma, label, color in zip(gammas_OLS, labels_gamma, colors):
    hist, n_gamma = sgd(X, y, n_epochs=100, batch_size=5, gamma=gamma, lmbda = lam)
    cost_gamma = cost_history(hist, X, y, penalty = "None")

    plt.plot(cost_gamma, label = label, color = color)

"""Plot of cost as for different learning rates"""
plt.title("Cost function for different learning rates")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(FIG_DIR/"Part_h_cost_learningrates.png")
plt.show()


"""Studying varying batch sizes"""
batches = [10, 20, 30]
labels_M = [f"$M = {batches[0]}$", f"$M = {batches[1]}$", f"$M = {batches[2]}$"]

for M, label, color in zip(batches, labels_M, colors):
    hist, n_M = sgd(X, y, n_epochs=100, batch_size=M, gamma=gamma_max_OLS)
    cost_M = cost_history(hist, X, y, penalty = "None")
    
    plt.plot(cost_M, label = label, color = color)
    
"""Plot of cost as for different batch sizes"""
plt.title("Cost function for different batch sizes")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(FIG_DIR/"Part_h_cost_batchsize.png")
plt.show()


"""Gradient descent for OLS and Ridge"""
#the largest safe learning rate for plain gradient descent: gamma < 2 / lambda_max(hessian)
H_OLS = hessian_eigs(X)
H_Ridge = hessian_eigs(X, lmbda= lam)

gamma_max_OLS = 2.0 / H_OLS.max()
gamma_max_Ridge = 2.0 / H_Ridge.max()

history_OLS, n_OLS = gradient_descent(X, y, gamma = gamma_max_OLS)
history_Ridge, n_Ridge = gradient_descent(X, y, lmbda = lam, gamma = gamma_max_Ridge)
cost_OLS = cost_history(history_OLS, X, y, penalty="None")
cost_Ridge = cost_history(history_Ridge, X, y, lmbda=lam, penalty="L2")

history_OLS_sgd, n_OLS_sgd = sgd(X, y )
history_Ridge_sgd, n_Ridge_sgd = sgd(X, y, lmbda = lam)
cost_OLS_sgd = cost_history(history_OLS_sgd, X, y, penalty = "None")
cost_Ridge_sgd = cost_history(history_Ridge_sgd, X, y, lmbda = lam, penalty = "L2")

print("OLS: GD vs. SGD------------------------------")
print(f'iterations to converge: GD: {n_OLS} SGD: {n_OLS_sgd}')
print(f'DG: theta = {history_OLS[-1]}')
print(f'SDG: theta = {history_OLS_sgd[-1]}')
print(f"theta: |sgd - gd| = {np.max(np.abs(history_OLS[-1] - history_OLS_sgd[-1])):.2e} ")


print("Ridge: GD vs. SGD------------------------------")
print(f'iterations to converge: GD: {n_Ridge} SGD: {n_Ridge_sgd}')
print(f'DG: theta = {history_Ridge[-1]}')
print(f'SDG: theta = {history_Ridge_sgd[-1]}')
print(f"theta: |sgd - gd| = {np.max(np.abs(history_Ridge[-1] - history_Ridge_sgd[-1])):.2e} ")


"""Plot of cost vs iteration"""
plt.plot(cost_OLS, label = "OLS GD", color = "#F433DA")
plt.plot(cost_OLS_sgd, label = "OLS SGD", color = "#F433DA", linestyle = "--", alpha = 0.4)
plt.plot(cost_Ridge, label = "Ridge GD", color = "#7326E6")
plt.plot(cost_Ridge_sgd, label = "Ridge SGD", color = "#7326E6", linestyle = "--", alpha = 0.4)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of stochastic gradient descent")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(FIG_DIR/"Part_h_sgd_OLS_Ridge.png")
plt.show()


"""Comparing train and test MSE across OLS, Ridge, and Lasso"""
n_train = X.shape[0]
n_test = X_test.shape[0]

theta_OLS_GD = history_OLS[-1]
theta_OLS_SGD= history_OLS_sgd[-1]
theta_Ridge_GD = history_Ridge[-1]
theta_Ridge_SGD= history_Ridge_sgd[-1]

methods = {
    "OLS (GD)": theta_OLS_GD,
    "OLS (SGD)": theta_OLS_SGD,
    "Ridge (GD)": theta_Ridge_GD,
    "Ridge (SGD)": theta_Ridge_SGD}

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
plt.savefig(FIG_DIR/"Part_h_hist_testtrain_sgd.png")
plt.show()