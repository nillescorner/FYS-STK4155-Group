"""
This file contain the neural network function used for project 2.

Feed forward, backward propagation, predict
"""

from Project_2.Code._Part_a import *


class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun = mse, cost_der = mse_der, lmbda = 0.0, reg = None):
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.lmbda = lmbda
        self.reg = reg
        self.layers = self.create_layers(network_input_size, layer_output_sizes)
 
    def create_layers(network_input_size, layer_output_size):
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
            W = jnp.random.randn(layer_output_size, i_size)
            b = jnp.random.randn(layer_output_size) 
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


    def cost(layers, inputs, activation_funcs, target, lmbda = 0.0, reg = None):
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
        
        predict = feed_forward(inputs, layers, activation_funcs)
        return mse(predict, target) + jnp.sum(penalty(W, lmbda, reg) for W, _ in layers)


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


    def backpropagation(inputs, layers, activation_funcs, targets, activation_ders, cost_der = mse_der, lmbda = 0.0, reg = None):
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

        layer_inputs, zs, predict = feed_forward_saver(inputs, layers, activation_funcs)
        layer_grads = [() for _ in layers]

        #loops over the layers, from last to first
        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
            if i == len(layers) - 1: 
                #last layer: cost derivative as dC/da(L)
                dC_da = cost_der(predict, targets)
            else: 
                #all other layers: build on previous z derivative, dCda(i) = dCdz(i + 1) * dz(i + 1)da(i)
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T
                
            dC_dz = dC_da * activation_der(z) 
            dC_dW = layer_input.T @ dC_dz + penalty_grad(W, lmbda, reg)
            dC_db = jnp.sum(dC_dz, axis = 0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads


