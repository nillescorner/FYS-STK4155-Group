"""
This python file contains the functions to do the linear regression methods OLS and Ridge based on the analytical
calculation of mean squared error and r2 score and theta.
"""

import numpy as np
from general_functions import design_matrix, scaling


def mse(y, y_tilde):
    """
    Analytical calculation of mean squared error

        Params:
            y (NDArray): target data
            y_tilde(NDArray): predicted data
        
        Returns:
            mse (float): mean squared error
    """
    return np.mean((y - y_tilde)**2)

def r2(y,y_tilde):
    """
    Analytical calculation of the r2 score

            Params:
            y (NDArray): target data
            y_tilde(NDArray): predicted data

            Returns:
            r2 (float): R2 score

    """
    return 1.0 - np.sum((y - y_tilde)**2) / np.sum((y - np.mean(y))**2)

def regression(x,y, lamba=0.0, mindeg = 1, maxdeg=16, seed=2026):
    """
    Regression method for either OLS (set lamba=0.0) or Ridge (set lamba = non-zero)
    Returns the mean squared error and r2 score for train and test data and the parameter theta
    
        Params:
            x (NDAraay): input data
            y (NDArray): input data
            lamba (float): penalty parameter, 0 = OLS
            mindegree (int): minimum degree (included)
            maxdegree (int): maximum degree (excluded)
            seed (int): to recreate same randomized data

        Returns:
            mse_train (list): mean squared error for training data for polynomials
            mse_test (list): mean squared error for test data for polynomials
            r2_train (list): r2 score for train data for polynomials
            r2_test (list): r2 score for test data for polynomials
            thetas (list): thetas for each degree 
    """
    mse_train, mse_test = [], []
    r2_train, r2_test = [], []
    thetas = []

    for deg in range(mindeg,maxdeg):
        X = design_matrix(x,degree=deg)
        #Scaled data using scaling function
        X_train, X_test, y_train, y_test = scaling(X,y, seed=seed, split_data=True) 
        theta = np.linalg.pinv(X_train.T @ X_train + lamba * np.eye(X_train.shape[1]) ) @ X_train.T @ y_train   #p85. lecturebook

        y_train_predict = X_train @ theta 
        y_test_predict = X_test @ theta

        mse_train.append(mse(y_train,y_train_predict))
        mse_test.append(mse(y_test,y_test_predict))

        r2_train.append(r2(y_train,y_train_predict))
        r2_test.append(r2(y_test,y_test_predict))

        thetas.append(theta)

    return mse_train, mse_test, r2_train, r2_test, thetas  
