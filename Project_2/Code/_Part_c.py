"""
This file contains the solution to task c, in project 2.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea

from _Part_a import Penalty, MSE, CrossEntropy, ReLU, LeakyReLU, Sigmoid, Identity
from optimizer import SGD, RMSprop, Adam, train
from neural_network import NeuralNetwork
from general_functions import RungeData, Scaler, FIG_DIR, save_best_lr, load_best_lr

np.random.seed(2026)

"""Getting and scaling the data"""
data = RungeData()

X_train, X_test, y_train, y_test = data.split(data.nn_features())
scaler = Scaler().fit(X_train, y_train)

X_train_s, y_train_s = scaler.transform(X_train, y_train)
X_test_s, y_test_s = scaler.transform(X_test, y_test)

y_train_s = y_train_s.reshape(-1, 1)
y_test_s = y_test_s.reshape(-1, 1)

'''
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
results_plot = np.where(np.isfinite(results), results, np.nan)
sea.heatmap(results_plot, xticklabels = [f"{g:.0e}" for g in gammas], yticklabels = list(ReLUs))
plt.xlabel(r"Learning rate $\gamma$")
plt.title("Test MSE, 1 hidden layer, 50 nodes")
plt.tight_layout()
plt.savefig(FIG_DIR / "Part_c_heatmaps_ReLUS.pdf")
plt.show()
'''

"""We now test different activation functions and depths of the nn using the best lrs for each"""
best_lrs_all = [load_best_lr(name)['best_lr'] for name in ('Sigmoid', 'ReLU', 'LeakyReLU')]
all_act_funcs ={'Sigmoid': Sigmoid(), 'ReLU': ReLU(), 'LeakyReLU': LeakyReLU()}
seeds = [21, 42, 67, 420, 2026]             #test for different seeds

batches=16

networks_sizes = {"1 x 50": [50, 1], "1 x 100": [100, 1], 
                  "2 x 50": [50, 50, 1], "2 x 100": [100, 100, 1],
                  '3 x 50': [50, 50, 50, 1], '3 x 100': [100, 100, 100, 1]}

train_mse = np.zeros((len(all_act_funcs), len(networks_sizes), len(seeds)))
test_mse = np.zeros_like(train_mse)

for i, (_, act) in enumerate(all_act_funcs.items()):
    gamma = best_lrs_all[i]
    for j, (__, sizes) in enumerate(networks_sizes.items()):
        activations = [act] * (len(sizes) - 1 ) + [Identity()]
        for k, seed in enumerate(seeds):
            nn = NeuralNetwork(1, sizes, activations, seed=seed)
            hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs=500, batch_size=batches, seed=seed, X_val = X_test_s, y_val = y_test_s)

            train_mse[i, j, k] = hist['train'][-1]
            test_mse[i, j, k] = hist['val'][-1]


train_mean, train_std = train_mse.mean(axis=2), train_mse.std(axis=2)
test_mean, test_std = test_mse.mean(axis=2), test_mse.std(axis=2)


labels = np.array([[rf"{m:.4f} $\pm$ {s:.4f}" for m, s in zip(row_m, row_s)]
                   for row_m, row_s in zip(test_mean, test_std)])

sea.heatmap(test_mean, annot=labels, fmt="", xticklabels=list(networks_sizes), yticklabels=list(all_act_funcs))
plt.title(rf"Test MSE, mean $\pm$ std over {len(seeds)} seeds")
plt.tight_layout()
plt.show()

for i, act_name in enumerate(all_act_funcs):
    for j, size_name in enumerate(networks_sizes):
        print(f'{act_name:10s} {size_name:10s}:'
              rf' Train MSE = {train_mean[i, j]:.4f} $\pm$ {train_std[i, j]:.4f}'
              rf' Test MSE = {test_mean[i, j]:.4f} $\pm$ {test_std[i,j]:.4f}')
