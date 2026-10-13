"""
This file contain the neural network function used for project 2.

Feed forward, backward propagation, predict
"""

from _Part_a import *
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp


def create_layers(network_input_size, layer_output_size, rng):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_size:
        W = rng.normal(0, 1 / jnp.sqrt(i_size), (i_size, layer_output_size))
        b = jnp.zeros(layer_output_size) + 0.01
        layers.append((W, b))
        i_size = layer_output_size
    return layers


def feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_func):
        a = activation_func(a @ W + b)
    return a


def feed_forward_saver(inputs, layers, activation_funcs):
    layer_inputs, zs = [], []
    a = inputs
    for (W, b), activation_funcs in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_funcs(z)
        zs.append(z)
    return layer_inputs, zs, a


def backpropagation(inputs, layers, activation_funcs, targets, activation_ders, cost_der = mse_der):
    layer_inputs, zs, predict = feed_forward_saver(inputs, layers, activation_funcs)
    layer_grads = [() for layer in layers]
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        if i == len(layers) - 1:
            dC_da = cost_der(predict, targets)
        else: 
            (W, b) = layers[i + 1]
            dC_da = dC_dz @ W.T
        dC_dz = dC_da * activation_der(z)
        dC_dW = layer_input.T @ dC_dz
        dC_db = jnp.sum(dC_dz, axis = 0)
        layer_grads[i] = (dC_dW, dC_db)
    return layer_grads
