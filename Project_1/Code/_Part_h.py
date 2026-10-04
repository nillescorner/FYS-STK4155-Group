"""
The file contains the code used to derive the plots of the cost functions for OLS and Ridge
regression using stochastic gradient descent (SGD), and compares SGD to plain (full-batch)
gradient descent for both methods.

The effect of learning rate and minibatch size on SGD's convergence is first studied separately,
sweeping over several learning rates (at a fixed batch size) and several batch sizes (at a fixed
learning rate), and plotting the resulting cost curves. These two sweeps are then repeated over a
much finer grid of values and shown as heatmaps of cost vs. epoch, which makes the trend across
many hyperparameter values visible at once instead of only the 3 discrete lines plotted earlier.

Plain gradient descent and SGD are then compared directly for OLS and Ridge, using the theoretical
maximum stable learning rate for gradient descent, to see how closely SGD's solution and iteration
count match those of full-batch gradient descent.

Train and test MSE are compared across the four resulting solutions (OLS and Ridge, each via GD and
SGD) to study generalisation.

The gradients were found using the gradient function, and gradient descent using the gradient_descent
function, both from gradient_descent_methods.py. The closed form solution is found using the
closed_form function, also from gradient_descent_methods.py. Minibatch stochastic gradient descent
is implemented in the sgd function, from sgd_methods.py.
"""

from sgd_methods import *
from general_functions import *
from gradient_descent_methods import *
from lasso_methods import *
from colors import *
from plots import *


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

theta_cf_OLS = closed_form(X, y)
theta_cf_Ridge = closed_form(X, y, lmbda = lam)
gamma_list = np.linspace(0.001, 0.4, 40)

"""the best gamma_values"""
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


"""Studying varying learning rate"""
gamma_max_OLS = 2.0 / np.linalg.eigvalsh(H_OLS).max()
gammas_OLS = [0.01 * gamma_max_OLS, 0.1 * gamma_max_OLS, 0.5 * gamma_max_OLS]
labels_gamma = [r"$0.01\, \gamma_{\max}$", r"$0.1\, \gamma_{\max}$", r"$0.5\, \gamma_{\max}$"]
colors_gamma = [LR_LOW, LR_MID, LR_HIGH]

for gamma, label, color in zip(gammas_OLS, labels_gamma, colors_gamma):
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
#plt.savefig(FIG_DIR/"Part_h_cost_learningrates.png")
plt.show()


"""Studying varying batch sizes"""
batches = [10, 20, 30]
labels_M = [f"$M = {batches[0]}$", f"$M = {batches[1]}$", f"$M = {batches[2]}$"]
colors_batch = [BATCH_SMALL, BATCH_MED, BATCH_LARGE]

for M, label, color in zip(batches, labels_M, colors_batch):
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
#plt.savefig(FIG_DIR/"Part_h_cost_batchsize.png")
plt.show()


"""Studying a decaying learning rate schedule: gamma_t = t0 / (t + t1) (Eq. 4.40)"""
# t1 controls how many steps pass before the schedule decays appreciably; t0 is chosen
# so the *initial* step gamma_1 = t0 / (1 + t1) matches each of the three constant
# learning rates used in "Studying varying learning rate" above, so this plot is
# directly comparable to that one - the only thing changing is letting gamma decay
# over training instead of holding it fixed.
t1_schedule = 50
schedules_OLS = [(gamma * (1 + t1_schedule), t1_schedule) for gamma in gammas_OLS]
 
for (t0, t1), label, color in zip(schedules_OLS, labels_gamma, colors_gamma):
    hist, n_sched = sgd(X, y, n_epochs=100, batch_size=5, schedule=(t0, t1), lmbda=lam)
    cost_sched = cost_history(hist, X, y, penalty="None")
 
    plt.plot(cost_sched, label=label, color=color)
 
"""Plot of cost for the decaying learning rate schedule"""
plt.title(r"Cost function for a decaying learning rate schedule $\gamma_t = t_0/(t+t_1)$")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig(FIG_DIR/"Part_h_cost_schedule.png")
plt.show()
 
 
"""Constant learning rate vs. decaying schedule, same initial step"""
gamma_const = gammas_OLS[1]  # the "0.1 gamma_max" rate used above
t0_match, t1_match = schedules_OLS[1]
 
hist_const, n_const = sgd(X, y, n_epochs=100, batch_size=5, gamma=gamma_const, lmbda=lam)
hist_sched, n_sched_match = sgd(X, y, n_epochs=100, batch_size=5, schedule=(t0_match, t1_match), lmbda=lam)
cost_const = cost_history(hist_const, X, y, penalty="None")
cost_sched_match = cost_history(hist_sched, X, y, penalty="None")
 
print("Constant gamma vs. decaying schedule (same initial step)----------------")
print(f"iterations to converge: constant = {n_const}  schedule = {n_sched_match}")
print(f"theta: |schedule - constant| = {np.max(np.abs(hist_sched[-1] - hist_const[-1])):.2e}")
 
plt.plot(cost_const, label=r"Constant $\gamma$", color=LR_MID)
plt.plot(cost_sched_match, label=r"Schedule $\gamma_t = t_0/(t+t_1)$", color=LR_MID, linestyle="--")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Constant learning rate vs. decaying schedule (same initial step)")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig(FIG_DIR/"Part_h_schedule_vs_constant.png")
plt.show()




"""Heatmap: varying learning rate (fixed batch_size = 5), fine log-spaced sweep.
Capped at 0.99*gamma_max_OLS rather than the exact boundary: gamma_max_OLS is the
theoretical limit for *full-batch* GD, and SGD's extra gradient noise on top of
that can tip rates right at or above the boundary into outright divergence
(see the gamma_max discussion in Part e) -- those show as inf/nan cost."""

gamma_grid = np.linspace(0.01 * gamma_max_OLS, 0.99 * gamma_max_OLS, 40)
cost_grid_gamma = cost_grid_over_param(X, y, param_name="gamma", param_values=gamma_grid,n_epochs=100, batch_size=5, lmbda=lam,)
plot_cost_heatmap(cost_grid_gamma, gamma_grid, r"Learning rate $\gamma$","Cost vs. epoch for varying learning rate (OLS, M = 5)", 
                  #fname=FIG_DIR/"Part_h_heatmap_learningrate.png",
                  relative_to=gamma_max_OLS)
plt.show()

"""Heatmap: varying batch size (fixed gamma = gamma_max_OLS), fine linear sweep"""
batch_grid = np.linspace(1, 100, 40).astype(int)
cost_grid_batch = cost_grid_over_param( X, y, param_name="batch_size", param_values=batch_grid, n_epochs=100, gamma=best_gamma_OLS,)
plot_cost_heatmap(cost_grid_batch, batch_grid, "Batch size $M$", r"Cost vs. epoch for varying batch size (OLS, $\gamma = \gamma_{best}$)",   
                  #fname=FIG_DIR/"Part_h_heatmap_batchsize.png", 
                  y_tick_fmt="{:.0f}",)
plt.show()


"""Gradient descent for OLS and Ridge"""
#the largest safe learning rate for plain gradient descent: gamma < 2 / lambda_max(hessian)
H_OLS = hessian_eigs(X)
H_Ridge = hessian_eigs(X, lmbda= lam)

gamma_max_OLS = 2.0 / H_OLS.max()
gamma_max_Ridge = 2.0 / H_Ridge.max()

history_OLS, n_OLS = gradient_descent(X, y, gamma = best_gamma_OLS)
history_Ridge, n_Ridge = gradient_descent(X, y, lmbda = lam, gamma = best_gamma_Ridge)
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


"""Comparing SGD optimisers: plain, momentum, Adagrad, RMSprop and Adam"""
beta_momentum = 0.9
opt_gammas = {"momentum": 0.1 * (1.0 - beta_momentum)} 

opt_methods = ["momentum", "adagrad", "rmsprop", "adam"]
opt_labels = {"plain": "Plain SGD", "momentum": "Momentum", "adagrad": "Adagrad",
              "rmsprop": "RMSprop", "adam": "Adam"}

opt_colors = {"plain": PLAIN, "momentum": MOMENTUM , "adagrad": ADAGRAD,
              "rmsprop": RMSPROP, "adam": ADAM}
 
histories_OLS_opt = {"plain": history_OLS_sgd}
histories_Ridge_opt = {"plain": history_Ridge_sgd}
n_OLS_opt = {"plain": n_OLS_sgd}
n_Ridge_opt = {"plain": n_Ridge_sgd}
 
for method in opt_methods:
    gamma_m = opt_gammas.get(method, 0.1)
    hist_OLS_m, n_OLS_m = sgd(X, y, method=method, gamma = gamma_m)
    hist_Ridge_m, n_Ridge_m = sgd(X, y, method=method, lmbda=lam, gamma = gamma_m)
    histories_OLS_opt[method], n_OLS_opt[method] = hist_OLS_m, n_OLS_m
    histories_Ridge_opt[method], n_Ridge_opt[method] = hist_Ridge_m, n_Ridge_m

theta_cf_OLS = closed_form(X, y)
iters_per_epoch_sgd = int(np.ceil(X.shape[0] / 5))  # default batch_size = 5
tol = 1e-2  # within 1% of the closed-form cost counts as "converged"
 
def iters_to_tolerance(history, cost_target, iters_per_epoch, tol=tol, **cost_kwargs):
    c = cost_history(history, X, y, **cost_kwargs)
    reached = np.flatnonzero(np.isfinite(c) & (c <= (1.0 + tol) * cost_target))
    return int(reached[0]) * iters_per_epoch if reached.size else None
 
cost_target_OLS = cost(theta_cf_OLS, X, y)
cost_target_Ridge = cost(theta_cf, X, y, lmbda=lam)

print(f"Optimiser comparison: iterations to reach within {tol:.0%} of the closed-form cost---")
for method in ["plain", *opt_methods]:
    it_OLS = iters_to_tolerance(histories_OLS_opt[method], cost_target_OLS, iters_per_epoch_sgd, penalty="None")
    it_Ridge = iters_to_tolerance(histories_Ridge_opt[method], cost_target_Ridge, iters_per_epoch_sgd, lmbda=lam, penalty="L2")
    it_OLS_str = f"{it_OLS:6d}" if it_OLS is not None else "   n/a"
    it_Ridge_str = f"{it_Ridge:6d}" if it_Ridge is not None else "   n/a"
    print(f"{opt_labels[method]:12s}: gamma = {opt_gammas.get(method, 0.1):<5} "
          f"OLS = {it_OLS_str}   Ridge = {it_Ridge_str}   "
          f"(out of {n_OLS_opt[method] - 1} total updates run)")

def _clipped_cost(history, *cost_args, cap_multiple=1e3, **cost_kwargs):
    c = cost_history(history, *cost_args, **cost_kwargs)
    baseline = c[0] if np.isfinite(c[0]) else np.nanmax(c[np.isfinite(c)])
    cap = cap_multiple * baseline
    return np.clip(np.where(np.isfinite(c), c, cap), None, cap)

 
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for method in ["plain", *opt_methods]:
    cost_OLS_m = _clipped_cost(histories_OLS_opt[method], X, y, penalty="None")
    cost_Ridge_m = _clipped_cost(histories_Ridge_opt[method], X, y, lmbda=lam, penalty="L2")
    axes[0].plot(cost_OLS_m, label=opt_labels[method], color=opt_colors[method])
    axes[1].plot(cost_Ridge_m, label=opt_labels[method], color=opt_colors[method])
 
axes[0].set_title("OLS")
axes[1].set_title("Ridge")
for ax in axes:
    ax.set_xlabel("Iteration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
axes[0].set_ylabel("Cost")
axes[0].legend()
fig.suptitle("Cost function for different SGD optimisers")
fig.tight_layout()
#fig.savefig(FIG_DIR/"Part_h_optimizer_comparison.png")
plt.show()



"""Plot of cost vs iteration"""
iters_per_epoch_sgd = int(np.ceil(X.shape[0] / 5))
x_OLS_sgd = np.arange(len(cost_OLS_sgd)) * iters_per_epoch_sgd
x_Ridge_sgd = np.arange(len(cost_Ridge_sgd)) * iters_per_epoch_sgd

plt.plot(cost_OLS, label = "OLS GD", color = OLS)
plt.plot(x_OLS_sgd, cost_OLS_sgd, label = "OLS SGD", color = OLS, linestyle = "--", alpha = 0.4)
plt.plot(cost_Ridge, label = "Ridge GD", color = RIDGE)
plt.plot(x_Ridge_sgd, cost_Ridge_sgd, label = "Ridge SGD", color = RIDGE, linestyle = "--", alpha = 0.4)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function of stochastic gradient descent")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.legend()
#plt.savefig(FIG_DIR/"Part_h_sgd_OLS_Ridge.png")
plt.show()


"""Comparing train and test MSE across OLS and Ridge"""
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
plt.bar(x - width/2, train_vals, width, label="Train MSE", color=TRAIN_DATA, alpha = 0.7)
plt.bar(x + width/2, test_vals, width, label="Test MSE", color=TEST_DATA, alpha = 0.7)
plt.xticks(x, labels, rotation=15)
plt.ylabel("MSE")
plt.title("Train vs. test MSE: OLS, Ridge, Lasso")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
#plt.savefig(FIG_DIR/"Part_h_hist_testtrain_sgd.png")
plt.show()