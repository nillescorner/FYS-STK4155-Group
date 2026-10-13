"""
This file contain the neural network function used for project 2.

Feed forward, backward propagation, predict
"""

from _Part_a import *
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp


def create_layers(network_input_size, layer_output_size, rng):
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

    for layer_output_size in layer_output_size:
        W = rng.normal(0, 1 / jnp.sqrt(i_size), (i_size, layer_output_size))
        b = jnp.zeros(layer_output_size) + 0.01
        layers.append((W, b))
        i_size = layer_output_size
    return layers


def feed_forward(input, layers, activation_funcs):
    """
    Feed-forward pass, a^l = f(a^(l-1) W^l + b^l), used for predictions
 
    Params:
        inputs (jnp.ndarray): input data, shape: (n, n_features)
        layers (list of (W, b)): weights and biases
        activation_funcs (list of callables): one activation function per layer
 
    Returns:
        jnp.ndarray: network output, shape: (n, n_outputs)
    """

    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        a = activation_func(a @ W + b)
    return a


def feed_forward_saver(inputs, layers, activation_funcs):
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

    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)
        zs.append(z)
    return layer_inputs, zs, a


def backpropagation(inputs, layers, activation_funcs, targets, activation_ders, cost_der = mse_der, lmbda = 0.0, reg = None, output_delta = None):
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
        output_delta (callable or None): if given, dC/dz^L = output_delta(predict, targets)
            is used directly for the output layer, default: None
 
    Returns:
        list of (dC_dW, dC_db): same structure as layers
    """

    layer_inputs, zs, predict = feed_forward_saver(inputs, layers, activation_funcs)
    layer_grads = [() for _ in layers]

    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        if i == len(layers) - 1:
            if output_delta is not None:
                dC_da = output_delta(predict, targets)
            else:
                dC_da = cost_der(predict, targets)
                dC_dz = dC_da * activation_der(z)
        else: 
            W_next, _ = layers[i + 1]
            dC_da = dC_dz @ W_next.T
            dC_dz = dC_da * activation_der(z) 

        W, _ = layers[i]
        dC_dW = layer_input.T @ dC_dz + penalty_grad(W, lmbda, reg)
        dC_db = jnp.sum(dC_dz, axis = 0)
        layer_grads[i] = (dC_dW, dC_db)
    return layer_grads


