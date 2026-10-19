"""
This file contains function for task a)

Cost functions, their derivatives and activations function and their derivatives, 
that will be used in project 2. 
"""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp


"""Penalty terms for the different regression methods (OLS, Ridge and Lasso), and their gradients"""
class Penalty:
    """
    Penalty term and its gradient for the different regression methods
 
    Params:
        reg (str or None): None for OLS, "L2" for Ridge, "L1" for Lasso
        lmbda (float): penalty strength
    """
 
    def __init__(self, reg=None, lmbda=0.0):
        if reg not in (None, "L1", "L2"):
            raise ValueError("reg must be None, L1 or L2")
        self.reg = reg
        self.lmbda = lmbda

 
    def __call__(self, theta):
        """
        L1 (Lasso): lambda * sum|theta_j|,  L2 (Ridge): lambda * sum theta_j^2,  OLS: 0
 
        Params:
            theta (jnp.ndarray): model parameters
 
        Returns:
            float: scalar penalty value (0 for OLS)
        """

        if self.reg is None or self.lmbda == 0: #OLS
            return 0.0
        
        if self.reg == "L2":    #Ridge
            return self.lmbda * jnp.sum(theta ** 2)
        
        return self.lmbda * jnp.sum(jnp.abs(theta)) #Lasso

 
    def grad(self, theta):
        """
        L1: lambda * sgn(theta),  L2: 2 * lambda * theta,  OLS: 0
 
        Params:
            theta (jnp.ndarray): model parameters
 
        Returns:
            jnp.ndarray: same shape as theta
        """

        if self.reg is None or self.lmbda == 0: # OLS
            return jnp.zeros_like(theta)
        
        if self.reg == "L2":    #Ridge
            return 2.0 * self.lmbda * theta
        
        return self.lmbda * jnp.sign(theta) #Lasso
      


"""Mean square error function, derivative, cost function and gradient of cost function"""
class MSE:
    """Mean squared error"""
 
    def __call__(self, predict, target):
        """
        Mean square error: C = (1/n) * sum (predict_i - target_i)^2
 
        Params:
            predict (jnp.ndarray): model predictions
            target (jnp.ndarray): true values, same shape as predict
 
        Returns:
            float: MSE
        """
        return jnp.mean((predict - target) ** 2)

 
    def der(self, predict, target):
        """
        Derivative of MSE: dC/dpredict = (2/n) * (predict - target)
 
        Params:
            predict (jnp.ndarray): model predictions
            target (jnp.ndarray): true values, same shape as predict
 
        Returns:
            jnp.ndarray: same shape as predict
        """
        return 2.0 * (predict - target) / predict.size

 
    def linear_cost(self, X, y, theta, penalty=Penalty()):
        """
        MSE cost function for a linear model X @ theta (project 1 regression)
 
        Params:
            X (jnp.ndarray): design matrix, shape: (n, p)
            y (jnp.ndarray): target values, shape: (n, )
            theta (jnp.ndarray): parameters, shape: (p, )
            penalty (Penalty): Penalty() for OLS, Penalty("L2", lmbda) for Ridge, Penalty("L1", lmbda) for Lasso
 
        Returns:
            float: value of cost function
        """
        return self(X @ theta, y) + penalty(theta)

 
    def linear_grad(self, X, y, theta, penalty=Penalty()):
        """
        Gradient of the MSE cost: (2/n) X^T (X theta - y) + penalty gradient
 
        Params:
            X (jnp.ndarray): design matrix, shape: (n, p)
            y (jnp.ndarray): target values, shape: (n, )
            theta (jnp.ndarray): parameters, shape: (p, )
            penalty (Penalty): Penalty() for OLS, Penalty("L2", lmbda) for Ridge, Penalty("L1", lmbda) for Lasso
 
        Returns:
            jnp.ndarray: the gradient, same shape as theta
        """
        n = X.shape[0]
        return 2.0 / n * X.T @ (X @ theta - y) + penalty.grad(theta)
 

"""Multiclass cross entropy (softmax)"""

class CrossEntropy:
    """Multiclass cross entropy, used together with a softmax output layer"""
 
    epsilon = 1e-12     #to avoid log(0)
 
    def __call__(self, P, Y):
        """
        Multiclass cross entropy: C = - sum_i sum_k (y_ik log(p_ik)) (eq. 5.28)
 
        Params:
            P (jnp.ndarray): predicted class probabilities, shape: (n, K)
            Y (jnp.ndarray): one-hot encoded targets, shape: (n, K)
 
        Returns:
            float: cross entropy
        """
        P = jnp.clip(P, self.epsilon, 1.0)
        return -jnp.sum(Y * jnp.log(P))
 
    def der(self, P, Y):
        """
        Derivative of multiclass cross entropy: dC/dp_ik = - y_ik / p_ik
 
        Params:
            P (jnp.ndarray): predicted class probabilities, shape: (n, K)
            Y (jnp.ndarray): one-hot encoded targets, shape: (n, K)
 
        Returns:
            jnp.ndarray: same shape as P
        """
        P = jnp.clip(P, self.epsilon, 1.0)
        return -Y / P
 
    def softmax_delta(self, P, Y):
        """
        Combined derivative of softmax + cross entropy: dC/dz = P - Y (eq. 8.69)
        Used directly as the output error delta^L when the output layer is softmax
 
        Params:
            P (jnp.ndarray): predicted class probabilities (softmax output), shape: (n, K)
            Y (jnp.ndarray): one-hot encoded targets, shape: (n, K)
 
        Returns:
            jnp.ndarray: same shape as P
        """
        return P - Y


"""Activation functions"""

class Identity:
    """
    Linear activation f(z) = z
    """

    def __call__(self, z):
        """
        Params:
        z (jnp.ndarray): input to activation function
        
        Returns: 
            jnp.ndarray: z unchanged
        """

        return z

    def der(self, z):
        """
        f'(z) = 1
        """

        return jnp.ones_like(z)

class Sigmoid:
    """Sigmoid activation function"""
 
    def __call__(self, z):
        """
        Sigmoid: sigma(z) = 1 / (1 + exp(-z))
 
        Params:
            z (jnp.ndarray): Input to activation function. z = a^(l - 1) W^l + b^l
 
        Returns:
            jnp.ndarray: same shape as z, every entry put to interval (0, 1)
        """
        return 1.0 / (1.0 + jnp.exp(-z))
 
    def der(self, z):
        """
        Derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z))
 
        Params:
            z (jnp.ndarray): Input to activation function. z = a^(l - 1) W^l + b^l
 
        Returns:
            jnp.ndarray: same shape as z, values in (0, 0.25]
        """
        s = self(z)
        return s * (1.0 - s)



class ReLU:
    """ReLU activation function """
 
    def __call__(self, z):
        """
        ReLU(z) = max(0, z) (eq. 8.23)
 
        Params:
            z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l
 
        Returns:
            jnp.ndarray: same shape as z, negative entries are set to 0, positive entries are held
        """
        return jnp.where(z > 0, z, 0.0)
 
    def der(self, z):
        """
        ReLU'(z) = 1 for z > 0, 0 for z <= 0 (eq. 8.23)
 
        Params:
            z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l
 
        Returns:
            jnp.ndarray: same shape as z, set to 1 for z > 0, and 0 for z <= 0, set to 0 at 0
        """
        return jnp.where(z > 0, 1.0, 0.0)
 

class LeakyReLU:
    """
    Leaky ReLU activation function
 
    Params:
        alpha (float): slope for negative inputs, default: 0.01
    """
 
    def __init__(self, alpha=0.01):
        self.alpha = alpha
 
    def __call__(self, z):
        """
        LReLU(z) = z for z > 0, alpha * z for z <= 0 (eq. 8.24)
 
        Params:
            z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l
 
        Returns:
            jnp.ndarray: same shape as z, negative entries are scaled by alpha, positive entries are held
        """
        return jnp.where(z > 0, z, self.alpha * z)

 
    def der(self, z):
        """
        LReLU'(z) = 1 for z > 0, alpha for z <= 0 (eq. 8.24)
 
        Params:
            z (jnp.ndarray): Input to the activation function. z = a^(l - 1) W^l + b^l
 
        Returns:
            jnp.ndarray: same shape as z, set to 1 for z > 0, and alpha for z <= 0, set to alpha at 0
        """
        return jnp.where(z > 0, 1.0, self.alpha)

class Softmax:
    """
    Softmax activation for the output layer in multiclass classification
    """

    def __call__(self, z):
        """
        Softmax, row-wise: p_ik = exp(z_ik) / sum_l exp(z_il)
        
        Params:
            z (jnp.ndarray): input to the output layer, shape: (n, K)
            
        Returns: 
            jnp.ndarray: same shape as z, each row is a probability
        """

        e = jnp.exp(z - jnp.max(z, axis = 1, keepdims = True)) #numerical stability

        return e / jnp.sum(e, axis = 1, keepdims = True)


 
