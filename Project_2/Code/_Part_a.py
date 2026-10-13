"""
This file contains function for task a)

Cost functions, their derivatives and activations function and their derivatives, 
that will be used in project 2. 
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp


"""Penalty terms for the different regression methods (OLS, Ridge and Lasso), and their gradients"""
def l1_penalty(theta, lmbda):
    """
    L1 penalty (Lasso): lambda * sum|theta_j|

    Params: 
        theta (jnp.ndarray): model parameters
        lmbda (float): penalty strength

    Returns: 
        float: scalar penalty value
    """

    return lmbda * jnp.sum(jnp.abs(theta))

def l2_penalty(theta, lmbda):
    """
    L2 penalty (Ridge): lambda * sum theta_j^2

    Params: 
        theta (jnp.ndarray): model parameters
        lmbda (float): penalty strength

    Returns: 
        float: scalar penalty value
    """
    return lmbda * jnp.sum(theta ** 2)

def penalty(theta, lmbda, reg):
    """
    Selects the penalty based on the regression method

    Params:
        theta (jnp.ndarray): model parameters
        lmbda (float): penalty strength
        reg (str or None): None for OLS, "L2" for Ridge, "L1" for Lasso

    Returns: 
        float: scalar penalty value (0 for OLS)
    """

    if reg is None or lmbda == 0: #for OLS
        return 0.0
    if reg == "L2": #for Ridge
        return l2_penalty(theta, lmbda)
    if reg == "L1": #for Lasso
        return l1_penalty(theta, lmbda)
    raise ValueError("reg must be None, L1 or L2")


def l1_penalty_grad(theta, lmbda):
    """
    Gradient for the L1 penalty: lambda * sgn(theta)

    Params: 
        theta (jnp.ndarray): model parameters
        lmbda (float): penalty strength

    Returns: 
        jnp.ndarray: same shape as theta
    """

    return lmbda * jnp.sign(theta)

def l2_penalty_grad(theta, lmbda):
    """
    Gradient for the L2 penalty: 2 * lambda * theta

    Params: 
        theta (jnp.ndarray): model parameters
        lmbda (float): penalty strength

    Returns: 
        jnp.ndarray: same shape as theta
    """
     
    return 2.0 * lmbda * theta

def penalty_grad(theta, lmbda, reg):
    """
    Selects the gradient of the penalty based on the regression method

    Params: 
        theta (jnp.ndarray): model parameters
        lmbda (float): penalty strength
        reg (str or None): None for OLS, "L2" for Ridge, "L1" for Lasso

    Returns:
        jnp.ndarray: same shape as theta
    """

    if reg is None or lmbda == 0: #for OLS
        return jnp.zeros_like(theta)
    if reg == "L2": #for Ridge
        return l2_penalty_grad(theta, lmbda)
    if reg == "L1": #for Lasso
        return l1_penalty_grad(theta, lmbda)
    raise ValueError("reg must be None, L1 or L2")
        

"""Mean square error function, derivative, cost function and gradient of cost function"""
def mse(predict, target):
    """
    Mean square error: C = (1/n) * sum (predict_i - target_i)^2 

    Params: 
        predict (jnp.ndarray): model predictions
        target (jnp.ndarray): true values 

    Returns:
        float: MSE
    """

    return jnp.mean((predict - target)**2)


def mse_der(predict, target):
    """
    Derivative of MSE: dC/dpredict = (2/n) * (predict - target)

    Params: 
        predict (jnp.ndarray): model predictions
        target (jnp.ndarray): true values 

    Returns:
        jnp.ndarray: same shape as predict
    """

    return 2.0*(predict - target)/predict.size 


def mse_cost(X, y, theta, lmbda = 0.0, reg = None):
    """
    MSE cost function for a linear model X @ theta

    Params: 
        X (jnp.ndarray): design matrix, shape: (n, p)
        y (jnp.ndarray): target values, shape: (n, )
        theta (jnp.ndarray): parameters, shape: (p, )
        lmbda (float): penalty strength
        reg (str or None): None for OLS, "L1" for Lasso, "L2" for Ridge

    Returns: 
        float: value of cost function
    """

    return mse(X @ theta, y) + penalty(theta, lmbda, reg)


def mse_grad(X, y, theta, lmbda = 0.0, reg = None):
    """
    Gradient of the MSE cost: (2/n) X^T (X theta - y) + (penalty for given regression method)

    Params: 
        X (jnp.ndarray): design matrix, shape: (n, p)
        y (jnp.ndarray): target values, shape: (n, )
        theta (jnp.ndarray): parameters, shape: (p, )
        lmbda (float): penalty strength
        reg (str or None): None for OLS, "L1" for Lasso, "L2" for Ridge

    Returns:
        jnp.ndarray: the gradient, same shape as theta
    """

    n = X.shape[0]
    return 2.0 / n * X.T @ (X @ theta - y) + penalty_grad(theta, lmbda, reg)


"""Multiclass cross entropy (softmax)"""
epsilon = 1e-12     #to avoid log(0)
def cross_entropy(P, Y):
    """
    multiclass cross entropy: C = - sum_i sum_k (y_ik log(p_ik)) (eq. 5.28)

    Params: 
        P (jnp.ndarray): predicted class probabilities, shape: (n, K)
        Y (jnp.ndarray): one-hot encoded targets, shape: (n, K)

    Returns: 
        float: cross entropy
    """

    P = jnp.clip(P, epsilon, 1.0)
    return - jnp.sum(Y * jnp.log(P)) 


def cross_entropy_der(P, Y):
    """
    derivative of multiclass cross entropy: dC/dp = - y_ik / p_ik

    Params: 
        P (jnp.ndarray): predicted class probabilities, shape: (n, K)
        Y (jnp.ndarray): one-hot encoded targets, shape: (n, K)

    Returns: 
        jnp.ndarray: same shape as P
    """

    P = jnp.clip (P, epsilon, 1.0)
    return - Y / P 

def softmax(z):
    """
    Softmax, row-wise: p_ik = exp(z_ik) / sum_l exp(z_il)

    Params:
        z (jnp.ndarray): Input to the output layer, shape: (n, K)

    Returns:
        jnp.ndarray: same shape as z, each row is a probability distribution (sums to 1)
    """

    e = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))   #subtract max for numerical stability
    return e / jnp.sum(e, axis=1, keepdims=True)


def softmax_cross_entropy_der(P, Y):
    """
    Derivative of cross entropy: dC/dz = P - Y

    Params:
        P (jnp.ndarray): predicted class probabilities (softmax output), shape: (n, K)
        Y (jnp.ndarray): one-hot encoded targets, shape: (n, K)

    Returns:
        jnp.ndarray: same shape as P
    """

    return P - Y



"""Activation functions"""
def sigmoid(z):
    """
    Sigmoid: sigma(z) = 1 / (1 + exp(-z))

    Params:
        z( jnp.ndarray): Input to activation function. z = a^(l - 1) W^l + b^l

    Returns: 
        jnp.ndarray: same shape as z,every entry put to interval (0, 1)
    """

    return 1.0/(1.0 + jnp.exp(-z))


def sigmoid_der(z):
    """
    derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z))

    Params: 
        z( jnp.ndarray): Input to activation function. z = a^(l - 1) W^l + b^l
        
    Returns: 
        jnp.ndarray: same shape as z 
    """
    
    s = sigmoid(z)
    return s*(1.0 - s)


def ReLU(z):
    """
    ReLU(z) = max(0, z) (eq. 8.23)

    Params:
        z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l 

    Returns: 
        jnp.ndarray: same shape as z, negative entries are set to 0, positive entries are held. 
    """

    return jnp.where(z > 0, z, 0.0)


def ReLU_der(z):
    """
    ReLU'(z) = 1 for z > 0, 0, for z <= 0 (eq. 8.23)

    Params:
        z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l 

    Returns: 
        jnp.ndarray: same shape as z, set to 1 for z > 0, and 0 for z <= 0, set to 0 at 0. 
    """

    return jnp.where(z > 0, 1.0, 0.0)


def leaky_ReLU(z, alpha = 0.01):
    """
    LReLU(z) = z for z > 0, alpha * z for z <= 0 (eq. 8.24)

    Params:
        z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l 
        alpha (float): slope for negative inputs, default: 0.01

    Returns: 
        jnp.ndarray: same shape as z, negative entries are scaled by alpha, positive entries are held. 
    """

    return jnp.where(z > 0, z, alpha * z)


def leaky_ReLU_der(z, alpha = 0.01):
    """
    LReLU'(z) = 1 for z > 0, alpha for z <= 0 (eq. 8.24)

    Params:
        z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l 
        alpha (float): slope for negative inputs, default: 0.01

    Returns: 
        jnp.ndarray: same shape as z, set to 1 for z > 0, and alpha for z <= 0, set to alpha at 0. 
    """

    return jnp.where(z > 0, 1.0, alpha)
