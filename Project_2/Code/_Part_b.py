"""
This file contains the solution to task b, in project 2.
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
from Project_2.Code._Part_a import Penalty, MSE, CrossEntropy, ReLU, LeakyReLU, Sigmoid
from Project_2.Code.optimizer import SGD, RMSprop, Adam
from Project_2.Code.neural_network import NeuralNetwork
from Project_2.Code.general_functions import RungeData

