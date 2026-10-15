"""
This python file contains the generalized functions used throughout this project.

The Runge function for our data, function for setting up the design matrix, 
making the x and y data, and scaling for data where it also can be split into
test and training data.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

def runge(x):
    """
    Runge function given in Project 1 description

        Params:
            x (NDArray or float): input values to generate Runge function
        
        Returns:
            f(x) (NDarray or float): The Runge function evaluated at input x
    """

    return 1.0 / (1.0 + 25.0 * x**2)


class RungeData:
    """
    Noisy data from the Runge function, y = f(x) + N(0, noise^2), x uniform in [-1, 1]
    Default at n = 100, noise = 0.1, seed = 2026
 
        Params:
            n (int): Number of data points
            noise (float): Noise on the data (sigma)
            seed (int): Use the same seed to generate same numbers 
 
        Attributes:
            x (NDArray): input data x, shape: (n,)
            y (NDArray): target data, shape: (n,)
    """

    def __init__(self, n = 100, noise = 0.1, seed = 2026):
        self.n, self.noise, self.seed = n, noise, seed
        rng = np.random.default_rng(seed)
        self.x = jnp.asarray(np.sort(rng.uniform(-1, 1, n)))
        self.y = runge(self.x) + jnp.asarray(rng.normal(0, noise, n))


    def design_matrix(x, degree, intercept=True):
        """
        Creates design matrix X
        Polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)

            Params:
                x (NDArray): input data
                degree (int): degree of polynomial
                intercept (bool): if False, column of ones is left out
            
            Returns:
                X (NDArray): design matrix
        """
        start = 0 if intercept else 1
        return jnp.vstack([x**p for p in range(start, degree + 1)]).T

    def polynomial_features(self, degree, intercept=True):
        """
        Design matrix of this data set 
        """

        return self.design_matrix(self.x, degree, intercept)
 
    def nn_features(self):
        """
        Raw x as a single input feature for the neural network, shape: (n, 1)
        """

        return self.x.reshape(-1, 1)
 
    def split(self, X, test_size=0.2):
        """
        Train-test split of a feature matrix X and this data's y, using the data's seed.
 
            Params:
                X (NDArray): features
                test_size (float): fraction of the data used for testing
 
            Returns:
                X_train, X_test, y_train, y_test (NDArray)
        """

        return train_test_split(X, self.y, test_size=test_size, random_state=self.seed)


class Scaler:
    """
    Normalizes a design matrix X and centers an array y, with the means and stds
    computed from the data it is fitted on 
 
    Based on p116 fit with intercept from lecturebook.
 
        Usage:
            scaler = Scaler().fit(X_train, y_train)
            X_train_s, y_train_s = scaler.transform(X_train, y_train)
            X_test_s, y_test_s = scaler.transform(X_test, y_test)
            y_pred = scaler.inverse_transform_y(model_output)   
    """
 
    def fit(self, X, y=None):
        """
        Computes the column means and stds of X, and the mean of y
 
            Params:
                X (NDArray): Design matrix
                y (NDArray or None): Target data
 
            Returns:
                self (Scaler): so that Scaler().fit(X, y) can be chained
        """

        self.X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self.X_std = jnp.where(X_std == 0, 1, X_std)   # leave constant columns alone
        self.y_mean = y.mean() if y is not None else 0.0
        return self
 
    def transform_X(self, X):
        """
        (X - X_mean) / X_std, with the fitted mean and std
        """
        return (X - self.X_mean) / self.X_std
 
    def transform_y(self, y):
        """
        y - y_mean, with the fitted mean
        """
        return y - self.y_mean
 
    def inverse_transform_y(self, y_scaled):
        """
        y_scaled + y_mean: brings predictions back to the original scale
        """
        return y_scaled + self.y_mean
 
    def transform(self, X, y):
        """
        Scales both X and y, 
        Returns: X_norm, y_centered
        """
        return self.transform_X(X), self.transform_y(y)
 
    def fit_transform(self, X, y):
        """
        Fits on X, y and scales them, 
        Returns: X_norm, y_centered
        """
        return self.fit(X, y).transform(X, y)



def scaling(X, y, seed=2026, split_data=False):
    """
    If split_data=True it returns X_train_norm, X_test_norm, y_train_centered, y_test_centered
    (test data scaled with the training mean/std), otherwise X_norm, y_centered
    """

    if split_data:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        scaler = Scaler().fit(X_train, y_train)
        return (scaler.transform_X(X_train), scaler.transform_X(X_test), scaler.transform_y(y_train), scaler.transform_y(y_test))
    
    return Scaler().fit_transform(X, y)
