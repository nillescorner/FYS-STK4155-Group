"""Generalized functions for program"""

import numpy as np
from sklearn.model_selection import train_test_split


def runge(x):
    """
    Runge function given in Project 1 description
    """
    return 1.0 / (1.0 + 25.0 * x**2)


def design_matrix(x, degree, intercept=True):
    """
    Creates design matrix X
    Polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)

        Params:
            x (any): input data
            degree (int): degree of polynomial
            intercept
    """
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T


def make_data(n=100, noise=0.1, seed=2026):
    """
    Creates data arrays x and y from number of data points, noise and seed.
    Default at n = 100, noise = 0.1, seed = 2026

        Parameters:
            n (int): Number of data points (observables)
            noise (float): Noise on the data (sigma)
            seed (int): Use the same seed to generate same numbers 
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n))
    y = runge(x) + rng.normal(0, noise, n)
    return x, y


def scaling(X,y, seed=2026, split_data=False):
    """
       Normalizes design matrix X and centers array y.
       Can be used with or without train_test_split by setting split_data = False. 
       If split_data= True it returns normalized design matrices: X_train, X_test and centered y_train, y_test
       If split_data = False it returns normalized design matrix X and centered array y.
   
       Based on p116 fit with intercept from lecturebook
       
           Params:
               X (NDArray): Design matrix X
               y (NDArray): y
               seed (int)
               split_data(bool): set to True if want to split into train and test data_points
    """
    if split_data == True:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed)

        X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train_std = np.where(X_train_std == 0, 1, X_train_std ) #leave constant columns alone
        y_train_mean = y_train.mean()

        #Scaling training data
        X_train_norm = (X_train - X_train_mean) / X_train_std
        y_train_centered = y_train - y_train_mean

        #Scaling testing data using training data to avoid data leakage
        X_test_norm = (X_test - X_train_mean) / X_train_std 
        y_test_centered = y_test - y_train_mean

        return X_train_norm, X_test_norm, y_train_centered, y_test_centered
    
    else:
        X_mean, X_std = X.mean(axis=0), X.std(axis=0)
        X_std = np.where(X_std == 0, 1, X_std ) #leave constant columns alone
        y_mean = y.mean()

        #Scale data
        X_norm = (X - X_mean) / X_std
        y_centered = y - y_mean

        return X_norm, y_centered
