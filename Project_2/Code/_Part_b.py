"""
This file contains the solution to task b, in project 2.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea

from Project_2.Code._Part_a import Penalty, MSE, CrossEntropy, ReLU, LeakyReLU, Sigmoid, Identity
from Project_2.Code.optimizer import SGD, RMSprop, Adam, train
from Project_2.Code.neural_network import NeuralNetwork
from Project_2.Code.general_functions import RungeData, Scaler, runge

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
plt.plot(x_grid, runge(x_grid), label = "Runge function")
plt.scatter(X_train.ravel(), y_train,  alpha = 0.6, label = "training data")
plt.scatter(X_test.ravel(), y_test, alpha = 0.6, label = "test data")
plt.plot(x_grid, y_grid, label = "neural network")
plt.plot(x_grid, y_grid_OLS, "--", label = f"OLS (degree {best_degree})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()


"""Plot of the MSE for train and test data"""
plt.plot(history50["train"], label = "Train")
plt.plot(history50["val"], label = "Test")
plt.xlabel("Epoch")
plt.ylabel("MSE")
#plt.xscale("log")
plt.yscale("log")
plt.title("MSE for training and test, with Adam")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


"""Learning rate sweep for different optimizers"""
gammas = np.logspace(-4, 0, 9)
batches = 16

optimizers = {"GD": lambda g: (SGD(g), None), "SGD": lambda g: (SGD(g), batches), "SGD + RMSprop": lambda g: (RMSprop(g), batches), "SGD + Adam": lambda g: (Adam(g), batches) }

results = np.zeros((len(optimizers), len(gammas)))

for i, (name, make) in enumerate(optimizers.items()):
    for j, gamma in enumerate(gammas):
        np.random.seed(2026)

        nn = NeuralNetwork(1, [50, 1], [Sigmoid(), Identity()])
        opt, batch_size = make(gamma)
        hist = train(nn, X_train_s, y_train_s, opt, epochs = 500, batch_size = batch_size, X_val = X_test_s, y_val = y_test_s)

        results[i, j] = hist["val"][-1]

        #print(f"{name:14s}, gamma = {gamma:.1e}, test MSE = {results[i,j]:.4f} ")


"""Finding the best learning rate"""
best = {}
for i, name in enumerate(optimizers):
    j = np.nanargmin(results[i])
    best[name] = (gammas[j], results[i, j])
    print(f"{name:14s}: best gamma = {gammas[j]:.1e}, MSE = {results[i, j]:.4f}")

i_best, j_best = np.unravel_index(np.nanargmin(results), results.shape)
best_name, best_gamma = list(optimizers)[i_best], gammas[j_best]

print(f"Best overall: {best_name} with gamma = {best_gamma:.1e}")


"""plotting heatmap for learningrate sweep"""
results_plot = np.where(np.isfinite(results), results, np.nan)
sea.heatmap(results_plot, xticklabels = [f"{g:.0e}" for g in gammas], yticklabels = list(optimizers))
plt.xlabel(r"Learning rate $\gamma$")
plt.title("Test MSE, 1 hidden layer, 50 nodes")
plt.tight_layout()
plt.show()


"""Different network sizes"""

networks_sizes = {"1 x 50": [50, 1], "1 x 100": [100, 1], "2 x 50": [50, 50, 1], "2 x 100": [100, 100, 1]}

networks, histories, y_grids = {}, {}, {}
for name, sizes in networks_sizes.items():
    np.random.seed(2026)

    activations = [Sigmoid()] * (len(sizes) - 1) + [Identity()]
    nn = NeuralNetwork(1, sizes, activations)

    print(f"{name:8s} gradient check: {nn.gradient_check(X_train_s, y_train_s):.2e}")

    hist = train(nn, X_train_s, y_train_s, Adam(gamma = 0.01), epochs=500, batch_size= batches, X_val = X_test_s, y_val = y_test_s)

    print(f"{name:.8s} train MSE: {hist['train'][-1]:.4f}, test MSE: {hist['val'][-1]:.4f}")

    networks[name], histories[name] = nn, hist
    y_grids[name] = scaler. inverse_transform_y(nn.feed_forward(x_grid_s)).ravel()
