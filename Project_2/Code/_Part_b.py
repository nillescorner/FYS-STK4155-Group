"""
This file contains the solution to task b, in project 2.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea
from matplotlib.colors import LogNorm

from _Part_a import Penalty, MSE, CrossEntropy, ReLU, LeakyReLU, Sigmoid, Identity
from optimizer import SGD, RMSprop, Adam, train
from neural_network import NeuralNetwork
from general_functions import RungeData, Scaler, runge, save_best_lr
from plot_colors import (COLORS, OPTIMIZER_COLORS, ARCH_COLORS, MSE_CMAP, TRAIN_LS, VAL_LS, set_style, plot_data_and_truth)
set_style()

np.random.seed(2026)

"""Getting and scaling the data"""
data = RungeData()

X_train, X_test, y_train, y_test = data.split(data.nn_features())
scaler = Scaler().fit(X_train, y_train)

X_train_s, y_train_s = scaler.transform(X_train, y_train)
X_test_s, y_test_s = scaler.transform(X_test, y_test)

y_train_s = y_train_s.reshape(-1, 1)
y_test_s = y_test_s.reshape(-1, 1)

"""Neural network training"""
nn50 = NeuralNetwork(1, [50, 1], [Sigmoid(), Identity()])
print(f'Gradient check (max |backprop - jax.grad|): {nn50.gradient_check(X_train_s, y_train_s):.2e}')

history50 = train(nn50, X_train_s, y_train_s, Adam(gamma = 0.01), epochs = 500, batch_size = 16, X_val = X_test_s, y_val = y_test_s)
print(f"Final train MSE: {history50["train"][-1]:.4f}, test MSE: {history50["val"][-1]:.4f} ")

x_grid = np.linspace(-1, 1, 400).reshape(-1, 1)
x_grid_s = scaler.transform_X(x_grid)                            
y_grid = scaler.inverse_transform_y(nn50.feed_forward(x_grid_s)).ravel()


"""Testing against OLS"""
best_degree = 8

X_polynom = data.polynomial_features(best_degree)
Xp_train, Xp_test, yp_train, yp_test = data.split(X_polynom)

scaler_OLS = Scaler().fit(Xp_train, yp_train)
Xp_train_s, yp_train_s = scaler_OLS.transform(Xp_train, yp_train)
Xp_test_s, yp_test_s = scaler_OLS.transform(Xp_test, yp_test)

theta_OLS = jnp.linalg.pinv(Xp_train_s.T @ Xp_train_s) @ Xp_train_s.T @ yp_train_s
MSE_OLS = MSE()(Xp_test_s @ theta_OLS, yp_test_s)
print(f"OLS (degree: {best_degree}) test MSE: {MSE_OLS:.4f}")

"""OLS predictions on the grid"""
X_grid_poly = RungeData.design_matrix(x_grid.ravel(), best_degree)
X_grid_poly_s = scaler_OLS.transform_X(X_grid_poly)
y_grid_OLS = scaler_OLS.inverse_transform_y(X_grid_poly_s @ theta_OLS)


"""Plotting the data, the runge function and the neural network data"""
fig, ax = plt.subplots()
plot_data_and_truth(ax, x_grid, runge(x_grid), X_train, y_train, X_test, y_test)
ax.plot(x_grid, y_grid, color = COLORS["nn"], label = "Neural network (1 x 50)")
ax.plot(x_grid, y_grid_OLS, "--", color = COLORS["ols"], label = f"OLS (degree {best_degree})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
fig.tight_layout()
#fig.savefig(f"{FIG_DIR}/Part_bfit_nn_vs_ols.pdf")
plt.show()


"""Plot of the MSE for train and test data"""
epochs50 = np.arange(1, len(history50["train"]) + 1)
fig, ax = plt.subplots()
ax.plot(epochs50, history50["train"], TRAIN_LS, color = COLORS["nn"], label = "Train")
ax.plot(epochs50, history50["val"], VAL_LS, color = COLORS["nn"], alpha = 0.5, label = "Test")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.set_title(r"MSE for training and test, Adam ($\gamma$ = 0.01)")
ax.legend()
fig.tight_layout()
#fig.savefig(f"{FIG_DIR}/Part_b_mse_train_test_adam.pdf")
plt.show()


"""Learning rate sweep for different optimizers"""
gammas = np.logspace(-4, 0, 9)
batches = 16

optimizers = {"GD": lambda g: (SGD(g), None), "SGD": lambda g: (SGD(g), batches), "SGD + RMSprop": lambda g: (RMSprop(g), batches), "SGD + Adam": lambda g: (Adam(g), batches) }
all_histories = {}

results = np.zeros((len(optimizers), len(gammas)))

for i, (name, make) in enumerate(optimizers.items()):
    for j, gamma in enumerate(gammas):
        np.random.seed(2026)

        nn = NeuralNetwork(1, [50, 1], [Sigmoid(), Identity()])
        opt, batch_size = make(gamma)
        hist = train(nn, X_train_s, y_train_s, opt, epochs = 500, batch_size = batch_size, X_val = X_test_s, y_val = y_test_s)

        all_histories[(name, j)] = hist
        results[i, j] = hist["val"][-1]
    
        #print(f"{name:14s}, gamma = {gamma:.1e}, test MSE = {results[i,j]:.4f} ")


"""Finding the best learning rate"""
best = {}
for i, name in enumerate(optimizers):
    j = np.nanargmin(results[i])
    best[name] = (gammas[j], results[i, j])
    print(f"{name:14s}: best gamma = {gammas[j]:.1e}, test MSE = {results[i, j]:.4f}")

i_best, j_best = np.unravel_index(np.nanargmin(results), results.shape)
best_name, best_gamma = list(optimizers)[i_best], gammas[j_best]

print(f"Best overall: {best_name} with gamma = {best_gamma:.1e}")


save_best_lr('Sigmoid', {'optimizer': best_name, 'best_lr': best_gamma, 'test_mse': results[i_best, j_best]})

"""plotting heatmap for learningrate sweep"""
results_plot = np.where(np.isfinite(results), results, np.nan)

fig, ax = plt.subplots(figsize = (9, 3.8))
sea.heatmap(results_plot, ax = ax, cmap = MSE_CMAP, norm = LogNorm(), annot = True, fmt = ".3f",
            annot_kws = {"size": 8}, linewidths = 1.8, linecolor = "white",
            xticklabels = [f"{g:.0e}" for g in gammas], yticklabels = list(optimizers),
            cbar_kws = {"label": "Test MSE"})
ax.grid(False)
ax.set_xlabel(r"Learning rate $\gamma$")
ax.set_title("Test MSE, 1 hidden layer, 50 nodes")
fig.tight_layout()
#fig.savefig(f"{FIG_DIR}/Part_b_heatmap_lr_optimizers.pdf")
plt.show()




"""The MSE for each optimizer, with the best gamma"""
for i, name in enumerate(optimizers):
    j = np.nanargmin(results[i])
    hist = all_histories[(name, j)]
    epochs = np.arange(1, len(hist["train"]) + 1)

    color = f"C{i}"
    plt.plot(epochs, hist["train"], "-", color=color, alpha = 0.4, label=rf"{name} ($\gamma$ = {gammas[j]:.0e})")
    plt.plot(epochs, hist["val"], "--", color=color)

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Solid: train, dashed: test")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize = (10, 7), sharex = True, sharey = True)
 
for ax, (i, name) in zip(axes.ravel(), enumerate(optimizers)):
    j = np.nanargmin(results[i])
    hist = all_histories[(name, j)]
    epochs = np.arange(1, len(hist["train"]) + 1)
    color = OPTIMIZER_COLORS[name]
 
    ax.plot(epochs, hist["train"], TRAIN_LS, color = color, label = "Train")
    ax.plot(epochs, hist["val"], VAL_LS, color = color, alpha = 0.8, label = "Test")
    ax.set_title(rf"{name}, $\gamma$ = {gammas[j]:.0e}")
    ax.set_yscale("log")
    ax.legend(loc = "upper right")
 
for ax in axes[1]:
    ax.set_xlabel("Epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("MSE")
 
fig.suptitle("Convergence at the best learning rate for each optimizer")
fig.tight_layout()
#fig.savefig(f"{FIG_DIR}/Part_b_convergence_optimizers.pdf")
plt.show()


"""Different network sizes"""
networks_sizes = {"1 x 50": [50, 1], "1 x 100": [100, 1], "2 x 50": [50, 50, 1], "2 x 100": [100, 100, 1]}

networks, histories, y_grids = {}, {}, {}
for name, sizes in networks_sizes.items():
    np.random.seed(2026)

    activations = [Sigmoid()] * (len(sizes) - 1) + [Identity()]
    nn = NeuralNetwork(1, sizes, activations)

    print(f"{name}------------------------------------")
    print(f"gradient check (max |backprop - jax.grad|): {nn.gradient_check(X_train_s, y_train_s):.2e}")

    hist = train(nn, X_train_s, y_train_s, Adam(gamma = best_gamma), epochs=500, batch_size= batches, X_val = X_test_s, y_val = y_test_s)

    print(f"train MSE: {hist['train'][-1]:.4f}, test MSE: {hist['val'][-1]:.4f}")

    networks[name], histories[name] = nn, hist
    y_grids[name] = scaler. inverse_transform_y(nn.feed_forward(x_grid_s)).ravel()


"""plot of the neural network fit for the different network sizes"""
fig, ax = plt.subplots()
plot_data_and_truth(ax, x_grid, runge(x_grid), X_train, y_train, X_test, y_test)
 
for name, y_g in y_grids.items():
    ax.plot(x_grid, y_g, color = ARCH_COLORS[name], label = name)
 
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
fig.tight_layout()
#fig.savefig(f"{FIG_DIR}/Part_b_fit_network_sizes.pdf")
plt.show()

"""plot of the MSE for the different network sizes"""
fig, ax = plt.subplots()
for name, hist in histories.items():
    epochs = np.arange(1, len(hist["val"]) + 1)
    ax.plot(epochs, hist["val"], color = ARCH_COLORS[name], label = name)
 
ax.set_xlabel("Epoch")
ax.set_ylabel("Test MSE")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
#fig.savefig(f"{FIG_DIR}/Part_b_mse_network_sizes.pdf")
plt.show()


