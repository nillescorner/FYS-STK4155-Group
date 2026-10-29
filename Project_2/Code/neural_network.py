"""
This file contain the neural network function used for project 2.

Feed forward, backward propagation, predict
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
from _Part_a import Penalty, MSE, MulticlassCrossEntropy, ReLU, LeakyReLU, Sigmoid, Softmax
from optimizer import zeros_like_grads

class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, activations, cost = MSE(), penalty = Penalty(), seed = 2026):
        self.activations = activations
        self.cost_obj = cost
        self.penalty = penalty
        self.layers = self.create_layers(network_input_size, layer_output_sizes, np.random.default_rng(seed))

    @staticmethod
    def create_layers(network_input_size, layer_output_sizes, rng):
        """
        Initilizes weights and bias for each layer

        Params:
            network_input_size (int): number of input features
            layer_output_sizes (list of int): number of nodes in each layer, the last one is the output layer
            rng (np.random.Generator): random number generator
    
        Returns:
            list of (W, b): W has shape (n_in, n_out), b has shape (n_out,)
        """

        layers = []
        i_size = network_input_size

        for layer_output_size in layer_output_sizes:
            W = jnp.asarray(rng.normal(0, 1 / np.sqrt(i_size), (i_size, layer_output_size)))
            b = jnp.zeros(layer_output_size) + 0.01
            layers.append((W, b))

            i_size = layer_output_size
        return layers


    def feed_forward(self, inputs):
        """
        Feed-forward pass, a^l = f(a^(l-1) W^l + b^l), used for predictions
    
        Params:
            inputs (jnp.ndarray): input data, shape: (n, n_features)
            layers (list of (W, b)): weights and biases
            activation_funcs (list of callables): one activation function per layer
    
        Returns:
            jnp.ndarray: network output, shape: (n, n_outputs)
        """

        return self.jax_compliant_predict(self.layers, inputs)


    def cost(self, inputs, targets):
        """
        Total cost of network

        Params: 
            layers (list of (W, b)): weights and bias
            inputs (jnp.ndarray): input data, shape: (n, n_features)
            activation_funcs (list of callables): one activation function per layer
            targets (jnp.ndarray): targets, same shape as outpu
            lmbda (float): penalty strength, default: 0.0
            reg (str or None): None,  "L1", "L2"
        """
        
        return self.jax_cost(self.layers, inputs, targets)


    def feed_forward_saver(self, inputs):
        """
        Feed-forward pass that also stores the inputs and z of every layer
    
        Params:
            inputs (jnp.ndarray): input data, shape: (n, n_features)
            layers (list of (W, b)): weights and biases
            activation_funcs (list of callables): one activation function per layer
    
        Returns:
            layer_inputs (list): a^(l-1) for every layer
            zs (list): z^l for every layer
            a (jnp.ndarray): network output, shape: (n, n_outputs)
        """

        layer_inputs, zs = [], []
        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activations):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a


    def backpropagation(self, inputs, targets):
        """
        Backpropagation: gradients of the cost wrt. every W and b
    
        Params:
            inputs (jnp.ndarray): input data, shape: (n, n_features)
            layers (list of (W, b)): weights and biases
            activation_funcs (list of callables): one activation function per layer
            targets (jnp.ndarray): targets, shape (n, n_outputs), same shape as the output
            activation_ders (list of callables): derivatives of the activation functions
            cost_der (callable): dC/da^L, default: mse_der
            lmbda (float): penalty strength
            reg (str or None): None, "L1" or "L2" (penalty on the weights only), default: None

        Returns:
            list of (dC_dW, dC_db): same structure as layers
        """

        layer_inputs, zs, predict = self.feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]

        #loops over the layers, from last to first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation = layer_inputs[i], zs[i], self.activations[i]
            if i == len(self.layers) - 1: 
                #last layer: cost derivative as dC/da(L)
                dC_da = self.cost_obj.der(predict, targets)
            else: 
                #all other layers: build on previous z derivative, dCda(i) = dCdz(i + 1) * dz(i + 1)da(i)
                W_next, b = self.layers[i + 1]
                dC_da = dC_dz @ W_next.T

            if i == len(self.layers) - 1 and isinstance(activation, Softmax):
                #special case: softmax has no der. using MulticlassCrossEntropy.softmax_delta
                dC_dz = self.cost_obj.softmax_delta(predict, targets)
            else: 
                #all other layers: dC_dz defined by derivative of activation funct.
                dC_dz = dC_da * activation.der(z)

            W, b = self.layers[i]
            dC_dW = layer_input.T @ dC_dz + self.penalty.grad(W)
            dC_db = jnp.sum(dC_dz, axis = 0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(self, layer_steps):
        """
        Applies the steps computed by the optimizer
        
        Params:
            layer_steps (list of (dW, db)): same stucture as layers 
        """

        self.layers = [(W - dW, b - db) for (W, b), (dW, db) in zip(self.layers, layer_steps)]


    #the next functions are to check the backproagation to JAX
    def jax_compliant_predict(self, layers, inputs):
        a = inputs
        for (W, b), activation_func in zip(layers, self.activations):
            a = activation_func(a @ W + b)
        return a


    def jax_cost(self, layers, inputs, targets):
        predict = self.jax_compliant_predict(layers, inputs)
        return self.cost_obj(predict, targets) + sum(self.penalty(W) for W, _ in layers)


    def jax_gradient(self, inputs, targets):
        """
        Gradient of the cost with jax.grad, same structure as compute_gradient
        """

        return jax.grad(self.jax_cost)(self.layers, inputs, targets)

    def gradient_check(self, inputs, targets):
        """
        finds the largest absolute difference between backpropagation and the jax_gradient
        """

        ours = self.backpropagation(inputs, targets)
        auto = self.jax_gradient(inputs, targets)
        return max(float(jnp.max(jnp.abs(g1 - g2))) for (dW1, db1), (dW2, db2) in zip (ours, auto) for g1, g2 in ((dW1, dW2), (db1, db2)))


    def train(
        self, X, y, optimizer, epochs=150, batch_size=None, seed=2026,
        X_val=None, y_val=None, eval_every=1):
        
        """Train this network and return the training/validation cost history."""
        optimizer_states = {
            "SGD": ("v",),
            "RMSprop": ("s",),
            "Adam": ("m", "v", "t"),
        }

        n = X.shape[0]
        if n == 0:
            raise ValueError("Cannot train on an empty dataset")
        if batch_size is None:
            batch_size = n
        elif batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be provided or both be None")

        keys = optimizer_states[type(optimizer).__name__]
        rng = np.random.default_rng(seed)
        optimizer.reset()

        zeros = zeros_like_grads(self.layers)
        state = tuple(
            jnp.asarray(0) if key == "t" else zeros
            for key in keys
        )
        history = {"train": [], "val": [], "epoch": []}

        def step(layers, state, X_batch, y_batch):
            self.layers = layers
            for key, value in zip(keys, state):
                setattr(optimizer, key, value)

            updates = optimizer.step(self.backpropagation(X_batch, y_batch))
            new_layers = [
                (W - dW, b - db)
                for (W, b), (dW, db) in zip(layers, updates)
            ]
            new_state = tuple(getattr(optimizer, key) for key in keys)
            return new_layers, new_state

        step = jax.jit(step)
        cost = jax.jit(self.jax_cost)

        layers = self.layers
        X_np, y_np = np.asarray(X), np.asarray(y)

        for epoch in range(epochs):
            indices = rng.permutation(n)

            for start in range(0, n, batch_size):
                batch_indices = indices[start:start + batch_size]
                layers, state = step(
                    layers,
                    state,
                    X_np[batch_indices],
                    y_np[batch_indices],
                )

            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                history["epoch"].append(epoch + 1)
                history["train"].append(float(cost(layers, X, y)))

                if X_val is not None:
                    history["val"].append(float(cost(layers, X_val, y_val)))

        self.layers = layers
        optimizer.reset()
        return history