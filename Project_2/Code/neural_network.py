"""
This file contain the neural network function used for project 2.

Feed forward, backward propagation, predict
"""
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
from Project_2.Code._Part_a import Penalty, MSE, CrossEntropy, ReLU, LeakyReLU, Sigmoid


class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, activations, cost = MSE(), penalty = Penalty()):
        self.activations = activations
        self.cost_obj = cost
        self.penalty = penalty
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
        
        return self._jax_cost(self.layers, inputs, targets)


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
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activations[i]
            if i == len(self.layers) - 1: 
                #last layer: cost derivative as dC/da(L)
                dC_da = self.cost_obj(predict, targets)
            else: 
                #all other layers: build on previous z derivative, dCda(i) = dCdz(i + 1) * dz(i + 1)da(i)
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T
                
            dC_dz = dC_da * activation_der(z) 
            dC_dW = layer_input.T @ dC_dz + self.penalty_grad(W)
            dC_db = jnp.sum(dC_dz, axis = 0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads


    #the next functions are to check the backproagation to JAX
    def jax_compliant_predict(self, layers, inputs):
        a = inputs
        for (W, b), activation_func in zip(layers, self.activations):
            a = activation_func(a @ W + b)
        return a


    def jax_cost(self, layers, inputs, targets):
        predict = self.jax_compliant_predict(layers, inputs)
        return self.cost_obj(predict, targets) + jnp.sum(self.penalty(W) for W, _ in layers)


    def jax_gradient(self, inputs, targets):
        """
        Gradient of the cost with jax.grad, same structure as compute_gradient
        """

        return jax.grad(self._jax_cost)(self.layers, inputs, targets)
        



