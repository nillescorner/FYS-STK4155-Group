"""
This file contains the solution to task c, in project 2.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea
from plot_colors import *
set_style()

from _Part_a import ReLU, LeakyReLU, Sigmoid, Identity
from optimizer import Adam, train
from neural_network import NeuralNetwork
from general_functions import RungeData, Scaler, FIG_DIR, save_best_lr, load_best_lr, ArgsHelper
import time

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
        hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs=500, batch_size=batches, X_val = X_test_s, y_val = y_test_s)
        results[i, j] = hist['val'][-1]


#Finding the best learning rate
best = {}
for i, act_func in enumerate(ReLUs):
    j = np.nanargmin(results[i])
    best[act_func] = (gammas[j], results[i, j])
    save_best_lr(act_func, {'optimizer': 'SGD + Adam', 'best_lr': gammas[j], 'test_mse': results[i,j]})
    print(f"{act_func:14s}: best gamma = {gammas[j]:.1e}, test MSE = {results[i, j]:.4f}")


#Plotting heatmap for learningrate sweep
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


"""We now test different activation functions and depths of the nn using the best lrs for each
activation function"""
all_act_funcs ={'Sigmoid': Sigmoid(), 'ReLU': ReLU(), 'LeakyReLU': LeakyReLU()}
best_lrs_all = [load_best_lr(name)['best_lr'] for name in all_act_funcs]
seeds = [21, 36, 67, 420, 2026]             #test for different seeds

n_layers = [1, 2, 3, 4]
n_nodes = [5, 25, 50, 100]
epochs = 200

#Keep same format (e.g '1 x 50': [50, 1]) but for all layers and nodes
networks_sizes = {f' {layers} x {nodes}': [nodes] * layers + [1] for layers in n_layers for nodes in n_nodes }

train_mse = np.zeros((len(all_act_funcs), len(networks_sizes), len(seeds)))
test_mse = np.zeros_like(train_mse)
fit_time = np.zeros_like(train_mse)

test_curves = np.zeros((len(all_act_funcs), len(networks_sizes), len(seeds), epochs))


for i, (_, act) in enumerate(all_act_funcs.items()):
    gamma = best_lrs_all[i]
    for j, (__, sizes) in enumerate(networks_sizes.items()):
        activations = [act] * (len(sizes) - 1 ) + [Identity()]
        for k, seed in enumerate(seeds):
            nn = NeuralNetwork(1, sizes, activations, seed=seed)

            time_start = time.perf_counter()
            hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs=epochs, batch_size=batches, seed=seed, X_val = X_test_s, y_val = y_test_s)
            time_end = time.perf_counter()

            train_mse[i, j, k] = hist['train'][-1]
            test_mse[i, j, k] = hist['val'][-1]
            fit_time[i, j, k] = time_end - time_start
            test_curves[i, j, k] = hist['val']

train_mean, train_std = train_mse.mean(axis=2), train_mse.std(axis=2)
test_mean, test_std = test_mse.mean(axis=2), test_mse.std(axis=2)

#reshape into 4x4 matrix 
shape = (len(all_act_funcs), len(n_layers), len(n_nodes))
test_mean_grid = test_mean.reshape(shape)

fig, axes = plt.subplots(1, len(all_act_funcs), figsize=(12, 5), sharey=True)

for i, (ax, name) in enumerate(zip(axes, all_act_funcs)):
    sea.heatmap(test_mean_grid[i], ax=ax, cmap=MSE_CMAP, annot=True, fmt='.3f',
            annot_kws = {"size": 8}, linewidths = 1.8, linecolor = "white",
            xticklabels = n_nodes, yticklabels = n_layers,
            cbar=(i == len(all_act_funcs) - 1), cbar_kws={"label": "Test MSE"})
    ax.set_title(name)
    ax.set_xlabel('Nodes per hidden layer')
    ax.grid(False)

axes[0].set_ylabel('Hidden layers')
fig.suptitle(f'Test MSE, mean over {len(seeds)} seeds for 3 activation functions for {epochs} epochs')
fig.tight_layout()
argshelper.plot_or_save('Part_c_heatmap_mse.pdf')

#Print every train and test MSE and the training time for each activation 
# function for each hidden layer and number of nodes 
mean_curves = test_curves.mean(axis=2)
time_median = np.median(fit_time, axis=2)
time_std = fit_time.std(axis=2)

for i, act_name in enumerate(all_act_funcs):
    for j, size_name in enumerate(networks_sizes):
        curve = mean_curves[i, j]
        best_epoch = np.argmin(curve)
        print(f'{act_name:10s} {size_name:10s}:'
              rf' Train MSE = {train_mean[i, j]:.4f} $\pm$ {train_std[i, j]:.4f}'
              rf' Test MSE = {test_mean[i, j]:.4f} $\pm$ {test_std[i, j]:.4f}',
              rf' Time = {time_median[i, j]:.3f} $\pm {time_std[i, j]:.3f}',
              rf' Best epoch {best_epoch + 1}')

#Heatmap for how long it takes for each network size for each activation function

time_grid = time_median.reshape(shape)

fig, axes = plt.subplots(1, len(all_act_funcs), figsize=(12, 5), sharey=True)
for i, (ax, name) in enumerate(zip(axes, all_act_funcs)):
    sea.heatmap(time_grid[i], ax=ax, cmap=MSE_CMAP, annot=True, fmt='.2f', 
                annot_kws={"size": 8}, linewidths=1.8, linecolor="white",
                xticklabels=n_nodes, yticklabels=n_layers,
                cbar=(i == len(all_act_funcs) - 1), cbar_kws={"label": "Time [s]"})
    ax.set_title(name)
    ax.set_xlabel('Nodes per hidden layer')
    ax.grid(False)
axes[0].set_ylabel('Hidden layers')
fig.suptitle(f'Median training time over {len(seeds)} seeds for 3 different activation functions for {epochs} epochs')
fig.tight_layout()
argshelper.plot_or_save('Part_c_heatmap_time.pdf')


"""We now want to check for overfitting, thus increase the number of epochs"""
big_epochs = 3000
eval_every = 25
big_network_size = [25, 25, 25, 25, 1]
n_points = big_epochs // eval_every

train_mse_long = np.zeros((len(all_act_funcs), n_points, len(seeds)))
test_mse_long = np.zeros_like(train_mse_long)

for i, (_, act) in enumerate(all_act_funcs.items()):
    gamma = best_lrs_all[i]
    activations = [act] * (len(big_network_size) - 1 ) + [Identity()]
    for k, seed in enumerate(seeds):
        nn = NeuralNetwork(1, big_network_size, activations, seed=seed)
        hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs=big_epochs, batch_size=batches, seed=seed, X_val = X_test_s, y_val = y_test_s, eval_every=eval_every)

        train_mse_long[i, :, k] = hist['train']
        test_mse_long[i, :, k] = hist['val']

long_ep = np.array(hist['epoch'])

"""Plot of train and test MSE against epoch, one row per activation"""
fig, axes = plt.subplots(len(all_act_funcs), 1, figsize=(7, 7), sharex=True, sharey=True)

for ax, (i, name) in zip(axes.ravel(), enumerate(all_act_funcs)):
    color = ACTIVATION_COLORS[name]
    train_mean_l, train_std_l = train_mse_long[i].mean(axis=-1), train_mse_long[i].std(axis=-1)
    test_mean_l, test_std_l = test_mse_long[i].mean(axis=-1), test_mse_long[i].std(axis=-1)

    ax.plot(long_ep, train_mean_l, TRAIN_LS, color=color, label='Train')
    ax.fill_between(long_ep, train_mean_l - train_std_l, train_mean_l + train_std_l, color=color, alpha=0.15)
    ax.plot(long_ep, test_mean_l, VAL_LS, color=color, label='Test')
    ax.fill_between(long_ep, test_mean_l - test_std_l, test_mean_l + test_std_l, color=color, alpha=0.15)
    ax.axhline(data.noise**2, color=COLORS['text'], ls=':', label=r'Noise floor $\sigma^2$')

    best_index = np.argmin(test_mean_l)     # mark the lowest test MSE
    print(f'{name:10s}: lowest test MSE {test_mean_l[best_index]:.4f} at epoch {long_ep[best_index]}, '
      f'At epoch {long_ep[-1]}: train MSE {train_mean_l[-1]:.4f}, test MSE{test_mean_l[-1]:.4f}')
    ax.plot(long_ep[best_index], test_mean_l[best_index], 'o', color=color, ms=4)

    ax.set_title(f'{name}, {len(big_network_size) - 1} x {big_network_size[0]}')    
    ax.set_ylabel('MSE')
    ax.set_yscale('log')

axes[0].legend(loc='upper right', ncol=3, fontsize=6.5)
fig.suptitle('Train and test MSE for longer epoch')
fig.tight_layout()
argshelper.plot_or_save('Part_c_long_epoch.pdf')

