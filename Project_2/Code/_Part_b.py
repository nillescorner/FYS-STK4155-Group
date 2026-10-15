"""
This file contains the solution to task b, in project 2.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from Project_2.Code._Part_a import Penalty, MSE, CrossEntropy, ReLU, LeakyReLU, Sigmoid, Identity
from Project_2.Code.optimizer import SGD, RMSprop, Adam, train
from Project_2.Code.neural_network import NeuralNetwork
from Project_2.Code.general_functions import RungeData, Scaler, runge


#getting data
data = RungeData()

X_train, X_test, y_train, y_test = data.split(data.nn_features())
scaler = Scaler().fit(X_train, y_train)

X_train_s, y_train_s = scaler.transform(X_train, y_train)
X_test_s, y_test_s = scaler.transform(X_test, y_test)

y_train_s = y_train_s.reshape(-1, 1)
y_test_s = y_test_s.reshape(-1, 1)

#neural network
nn = NeuralNetwork(1, [50, 1], [Sigmoid(), Identity()])
print(f'Gradient check (max |backprop - jax.grad|): {nn.gradient_check(X_train_s, y_train_s)}')

history = train(nn, X_train_s, y_train_s, Adam(gamma = 0.01), epochs = 500, batch_size = 16, X_val = X_test_s, y_val = y_test_s)
print(f"Final train MSE: {history["train"][-1]}, test MSE: {history["val"][-1]} ")

x_grid = np.linspace(-1, 1, 400).reshape(-1, 1)
x_grid_s = scaler.transform_X(x_grid)                            
y_grid = scaler.inverse_transform_y(nn.feed_forward(x_grid_s)).ravel()


plt.plot(x_grid, runge(x_grid), label = "Runge function")
plt.scatter(X_train.ravel(), y_train,  alpha = 0.6, label = "training data")
plt.scatter(X_test.ravel(), y_test, alpha = 0.6, label = "test data")
plt.plot(x_grid, y_grid, label = "neural network")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()










