"""
This file contains the solution to task c, in project 2.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea
from plot_colors import *
set_style()

from _Part_a import ReLU, LeakyReLU, Sigmoid, Identity
from optimizer import Adam, train
from neural_network import NeuralNetwork
from general_functions import RungeData, Scaler, FIG_DIR, save_best_lr, load_best_lr, ArgsHelper


np.random.seed(2026)
argshelper = ArgsHelper(fig_dir=FIG_DIR) # assists with parsed arguments if any

"""Getting and scaling the data"""
data = RungeData()

X_train, X_test, y_train, y_test = data.split(data.nn_features())
scaler = Scaler().fit(X_train, y_train)

X_train_s, y_train_s = scaler.transform(X_train, y_train)
X_test_s, y_test_s = scaler.transform(X_test, y_test)

y_train_s = y_train_s.reshape(-1, 1)
y_test_s = y_test_s.reshape(-1, 1)


"""Find the best learning rates for ReLU and LeakyReLU as done in part b using
the same seed (2026), same number of batches, nodes and hidden layers (16, 50 and 1)"""

gammas = np.logspace(-4, 0, 9)
batches = 16

ReLUs = {'ReLU': ReLU(), 'LeakyReLU': LeakyReLU()}
results = np.zeros((len(ReLUs), len(gammas)))

for i, (name, activation) in enumerate(ReLUs.items()):
    for j, gamma in enumerate(gammas):
        nn = NeuralNetwork(1, [50, 1], [activation, Identity()])
        hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs = 500, batch_size = batches, X_val = X_test_s, y_val = y_test_s)
        results[i, j] = hist['val'][-1]

"""Finding the best learning rate"""
best = {}
for i, act_func in enumerate(ReLUs):
    j = np.nanargmin(results[i])
    best[act_func] = (gammas[j], results[i, j])
    save_best_lr(act_func, {'Optimizer': 'SGD + Adam', 'best_lr': gammas[j], 'test_mse': results[i,j]})
    print(f"{act_func:14s}: best gamma = {gammas[j]:.1e}, test MSE = {results[i, j]:.4f}")


"""Plotting heatmap for learningrate sweep"""
fig, ax = plt.subplots(figsize = (6, 6))
results_plot = np.where(np.isfinite(results), results, np.nan)
sea.heatmap(results_plot, ax = ax, cmap = MSE_CMAP, annot = True, fmt = ".3f",
            annot_kws = {"size": 8}, linewidths = 1.8, linecolor = "white",
            xticklabels = [f"{g:.0e}" for g in gammas], yticklabels = list(ReLUs),
            cbar_kws = {"label": "Test MSE"})

ax.set_xlabel(r"Learning rate $\gamma$")
ax.grid(False)
ax.set_title("Test MSE, 1 hidden layer, 50 nodes")
fig.tight_layout()
argshelper.plot_or_save("Part_c_heatmaps_ReLUS.pdf")


"""We now test different activation functions and depths of the nn using the best lrs for each"""
best_lrs_all = [load_best_lr(name)['best_lr'] for name in ('Sigmoid', 'ReLU', 'LeakyReLU')]
all_act_funcs ={'Sigmoid': Sigmoid(), 'ReLU': ReLU(), 'LeakyReLU': LeakyReLU()}
seeds = [21, 36, 67, 420, 2026]             #test for different seeds

epochs = 150


networks_sizes = {"1 x 50": [50, 1], "1 x 100": [100, 1], '1 x 150': [150, 1], 
                  "2 x 50": [50, 50, 1], "2 x 100": [100, 100, 1], '2 x 150': [150, 150, 1],
                  '3 x 50': [50, 50, 50, 1], '3 x 100': [100, 100, 100, 1], '3 x 150': [150, 150, 150, 1],
                  '4 x 50': [50, 50, 50, 50, 1], '4 x 100': [100, 100, 100, 100, 1], '4 x 150': [150, 150, 150, 150, 1]}


train_mse = np.zeros((len(all_act_funcs), len(networks_sizes), len(seeds)))
test_mse = np.zeros_like(train_mse)

val_curves = np.zeros((len(all_act_funcs), len(networks_sizes), len(seeds), epochs))


for i, (_, act) in enumerate(all_act_funcs.items()):
    gamma = best_lrs_all[i]
    for j, (__, sizes) in enumerate(networks_sizes.items()):
        activations = [act] * (len(sizes) - 1 ) + [Identity()]
        for k, seed in enumerate(seeds):
            nn = NeuralNetwork(1, sizes, activations, seed=seed)
            hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs=epochs, batch_size=batches, seed=seed, X_val = X_test_s, y_val = y_test_s)

            train_mse[i, j, k] = hist['train'][-1]
            test_mse[i, j, k] = hist['val'][-1]
            val_curves[i, j, k] = hist['val']


train_mean, train_std = train_mse.mean(axis=2), train_mse.std(axis=2)
test_mean, test_std = test_mse.mean(axis=2), test_mse.std(axis=2)


labels = np.array([[f"{m:.3f} ± {s:.3f}" for m, s in zip(rm, rs)]
                   for rm, rs in zip(test_mean, test_std)])

fig, ax = plt.subplots(figsize = (7, 7))
sea.heatmap(test_mean, ax = ax, cmap=MSE_CMAP, annot = True, fmt = ".3f", 
             annot_kws = {"size": 8}, linewidths = 1.8, linecolor = "white",
             xticklabels=list(networks_sizes), yticklabels=list(all_act_funcs), cbar_kws={'label': 'Test MSE'})

ax.set_title(rf'Test MSE, mean $\pm$ std over {len(seeds)} seeds')
ax.grid(False)
fig.tight_layout()
argshelper.plot_or_save('Part_c_heatmap_networks.pdf')



"""Print every train and test MSE for each activation function for each hidden layer and number of nodes"""
mean_curves = val_curves.mean(axis=2)

for i, act_name in enumerate(all_act_funcs):
    for j, size_name in enumerate(networks_sizes):
        curve = mean_curves[i, j]
        best_epoch = np.argmin(curve)
        print(f'{act_name:10s} {size_name:10s}:'
              rf' Train MSE = {train_mean[i, j]:.4f} $\pm$ {train_std[i, j]:.4f}'
              rf' Test MSE = {test_mean[i, j]:.4f} $\pm$ {test_std[i,j]:.4f}',
              rf' Epoch {best_epoch + 1}')


"""We now want to check for overfitting, thus increase the number of epochs"""
big_epochs = 4000
eval_every = 25
big_network_size = [150, 150, 150, 150, 1]
n_ponts = big_epochs // eval_every

train_mse_long = np.zeros((len(all_act_funcs), n_ponts, len(seeds)))
test_mse_long = np.zeros_like(train_mse_long)


for i, (_, act) in enumerate(all_act_funcs.items()):
    gamma = best_lrs_all[i]
    activations = [act] * (len(big_network_size) - 1 ) + [Identity()]
    for k, seed in enumerate(seeds):
        nn = NeuralNetwork(1, big_network_size, activations, seed=seed)
        hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs=big_epochs, batch_size=batches, seed=seed, X_val = X_test_s, y_val = y_test_s, eval_every=eval_every)

        train_mse_long[i, :, k] = hist['train']
        test_mse_long[i, :, k] = hist['val']


long_ep = np.arange(1, n_ponts + 1) * eval_every


