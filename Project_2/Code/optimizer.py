"""
This code contains the optimizer methods we will use in project 2.

Gradient descent, stochastich gradient descent, RMSprop, adam.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp



def zeros_like_grads(layer_grads):
    return [(jnp.zeros_like(dW), jnp.zeros_like(db)) for dW, db in layer_grads]


class SGD:
    """
    (Stochastic) gradient descent
    """

    def __init__(self, gamma, momentum = 0.0):
        self.gamma, self.momentum = gamma, momentum
        self.reset()


    def reset(self):
        self.v = None


    def step(self, layer_grads):
        if self.v is None:
            self.v = zeros_like_grads(layer_grads)

        self.v = [(self.momentum * vW + self.gamma * dW, self.momentum * vb + self.gamma * db) for (dW, db), (vW, vb) in zip(layer_grads, self.v)]

        return self.v



class RMSprop:
    """
    RMSprop: s <- rho * s + (1 - rho) * g^2,   step = gamma * g / (sqrt(s) + eps)
    """

    def __init__(self, gamma, rho = 0.9, eps = 1e-8):
        self.gamma, self.rho, self.eps = gamma, rho, eps
        self.reset()

 
    def reset(self):
        self.s = None

 
    def step(self, layer_grads):
        if self.s is None:
            self.s = zeros_like_grads(layer_grads)

        self.s = [(self.rho * sW + (1 - self.rho) * dW**2, self.rho * sb + (1 - self.rho) * db**2) for (dW, db), (sW, sb) in zip(layer_grads, self.s)]

        return [(self.gamma * dW / (jnp.sqrt(sW) + self.eps), self.gamma * db / (jnp.sqrt(sb) + self.eps)) for (dW, db), (sW, sb) in zip(layer_grads, self.s)]
 


class Adam:
    """
    Adam: 
    m <- beta1 * m + (1 - beta1) * g,   
    v <- beta2 * v + (1 - beta2) * g^2,
    m_hat = m / (1 - beta1^t),   
    v_hat = v / (1 - beta2^t),
    step = gamma * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, gamma, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.gamma, self.beta1, self.beta2, self.eps = gamma, beta1, beta2, eps
        self.reset()


    def reset(self):
        self.m, self.v, self.t = None, None, 0


    def step(self, layer_grads):
        if self.m is None:
            self.m, self.v = zeros_like_grads(layer_grads), zeros_like_grads(layer_grads)
        self.t += 1

        b1, b2 = self.beta1, self.beta2

        self.m = [(b1 * mW + (1 - b1) * dW, b1 * mb + (1 - b1) * db) for (dW, db), (mW, mb) in zip(layer_grads, self.m)]
        self.v = [(b2 * vW + (1 - b2) * dW**2, b2 * vb + (1 - b2) * db**2) for (dW, db), (vW, vb) in zip(layer_grads, self.v)]

        c1, c2 = 1 - b1**self.t, 1 - b2**self.t

        return [(self.gamma * (mW / c1) / (jnp.sqrt(vW / c2) + self.eps), self.gamma * (mb / c1) / (jnp.sqrt(vb / c2) + self.eps)) for (mW, mb), (vW, vb) in zip(self.m, self.v)]


def train(network, X, y, optimizer, epochs = 500, batch_size = None, seed = 2026, X_val = None, y_val = None):
    """
    Trains the network

    Params: 
        network (NeuralNetwork): the network to train
        X, y (jnp.ndarray): training data
        optimizer (int): SGD, RMSprop or Adam
        epochs (int): number of passes over the training data
        batch_sizes (int or None): None: plain gradient descent, otherwise SGD with minibatches
        seed (int): seed for reshuffeling the data every epoch 
        X_val, y_val (jnp.ndarray or None): validation data, for cost history

    Returns:
        dict: cost per epoch
    """

    n = X.shape[0]
    batch_size = batch_size or n
    rng = np.random.default_rng(seed)
    optimizer.reset()
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start: start + batch_size]
            layer_grads = network.backpropagation(X[batch], y[batch])
            network.update_weights(optimizer.step(layer_grads))

        history["train"].append(float(network.cost(X, y)))
        if X_val is not None:
            history["val"].append(float(network.cost(X_val, y_val)))

    return history

