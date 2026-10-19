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
from general_functions import RungeData, Scaler, runge

np.random.seed(2026)

"""Getting and scaling the data"""
data = RungeData()

X_train, X_test, y_train, y_test = data.split(data.nn_features())
scaler = Scaler().fit(X_train, y_train)

X_train_s, y_train_s = scaler.transform(X_train, y_train)
X_test_s, y_test_s = scaler.transform(X_test, y_test)

y_train_s = y_train_s.reshape(-1, 1)
y_test_s = y_test_s.reshape(-1, 1)


"""Find the best learning rates for ReLU and LeakyReLU"""
gammas = np.logspace(-4, 0, 9)
batches = 16


ReLUs = {'ReLU': ReLU(), 'LeakyReLU': LeakyReLU()}
results = np.zeros((len(ReLUs), len(gammas)))
all_histories = {}



for i, (name, activation) in enumerate(ReLUs.items()):
    for j, gamma in enumerate(gammas):

        nn = NeuralNetwork(1, [50, 1], [activation, Identity()], seed=2026)
        hist = train(nn, X_train_s, y_train_s, Adam(gamma), epochs = 500, batch_size = batches, X_val = X_test_s, y_val = y_test_s)

        all_histories[(name, j)] = hist
        results[i, j] = hist['val'][-1]
    


"""Finding the best learning rate"""
best = {}
for i, act_func in enumerate(ReLUs):
    j = np.nanargmin(results[i])
    best[act_func] = (gammas[j], results[i, j])
    print(f"{act_func:14s}: best gamma = {gammas[j]:.1e}, test MSE = {results[i, j]:.4f}")



networks_sizes = {"1 x 50": [50, 1], "1 x 100": [100, 1], 
                  "2 x 50": [50, 50, 1], "2 x 100": [100, 100, 1],
                  '3 x 50': [50, 50, 50, 1], '3 x 100': [100, 100, 100, 1]}

