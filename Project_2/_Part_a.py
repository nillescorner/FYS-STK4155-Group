"""
This file contains function for task a)

Cost functions, their derivatives and activations function and their derivatives, 
that will be used in project 2. 

"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp


"""Penalty terms for the different regression methods, and their gradients"""
def l1_penalty(theta, lmbda):
    return lmbda * np.sum(np.abs(theta))

def l2_penalty(theta, lmbda):
    return lmbda * np.sum(theta ** 2)

def penalty(theta, lmbda, reg):
    if reg is None or lmbda == 0: 
        return 0.0
    if reg == "L2":
        return l2_penalty(theta, lmbda)
    if reg == "L1":
        return l1_penalty(theta, lmbda)
    raise ValueError("reg must be None, L1 or L2")


def l1_penalty_grad(theta, lmbda):
    return 2.0 * lmbda * theta

def l2_penalty_grad(theta, lmbda):
    return lmbda * np.sign(theta)

def penalty_grad(theta, lmbda, reg):
    if reg is None or lmbda == 0:
        return np.zeros_like(theta)
    if reg == "L2":
        return l2_penalty_grad(theta, lmbda)
    if reg == "L1":
        return l1_penalty_grad(theta, lmbda)
    raise ValueError("reg must be None, L1 or L2")
        

"""Mean square error function, derivative, cost function and gradient of cost function"""
def mse(predict, target):
    return jnp.mean((predict - target)**2)


def mse_der(predict, target):
    return 2.0*(predict - target)/predict.size 


def mse_cost(X, y, theta, lmbda = 0.0, reg = None):
    return mse(X @ theta, y) + penalty(theta, lmbda, reg)


def mse_grad(X, y, theta, lmbda = 0.0, reg = None):
    n = X.shape[0]
    return 2.0 / n * X.T @ (X @ theta - y) + penalty_grad(theta, lmbda, reg)


"""Multiclass cross entropy (softmax)"""
epsilon = 1e-12
def cross_entroy(P, Y):
    P = np.clip(P, epsilon, 1.0)
    return - np.sum(Y * np.log(P)) / (Y.shape[0])


def cross_entropy_der(P, Y):
    P = np.clip (P, epsilon, 1.0)
    return - Y / P / Y.shape[0]



"""Activation functions"""
def sigmoid(z):
    return 1.0/(1.0 + jnp.exp(-z))


def sigmoid_der(z):
    s = sigmoid(z)
    return s*(1.0 - s)


def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)


def leaky_ReLU(z, alpha = 0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_ReLU_der(z, alpha = 0.01):
    return np.where(z > 0, 1.0, alpha)
