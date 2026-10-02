"""
This python file contains the functions necessary to do the resampling techniques bootstrap and cross-validation
"""

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


def bootstrap_resampling(x,y, mindegree=1, maxdegree=21, n_bootstraps=100, seed=2026):
    """

    Bootstrap resampling function for simpler ordinary least squares based on code from p.65

        Params:
            x (any): input values
            y (any): input values
            mindegree (int): minimum degree
            maxdegree (int): maximum degree
            n_bootstraps (int): number of iterations for bootstrap
            seed (int): randomizer seed
        
        Returns:
            error (list): test error for each polynomial degree
            bias (list): bias for each polynomial degree
            variance (list): variance for each polynomial degree
    """                                
    x = x.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    error, bias, variance = [], [], []

    for deg in range(mindegree,maxdegree):
        #StandardScaler does X_norm, LinearRegression 
        model = make_pipeline(PolynomialFeatures(degree=deg),
                              StandardScaler(),
                              LinearRegression())

        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train, random_state= i+1)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

        error.append(np.mean(np.mean((y_test[:, None] - y_pred)**2, axis=1)))
        bias.append(np.mean((y_test[:, None] - np.mean(y_pred, axis=1, keepdims=True))**2))
        variance.append(np.mean(np.var(y_pred, axis=1)))

    return error, bias, variance


def cross_validation(x, y, model='ols', lamba=0.0, mindegree=1, maxdegree=21, k=5, max_iter=10000, seed=2026):
    """
    Cross-validation resampling technique. Works for OLS, Ridge, Lasso

        Params:
            x (any): input value
            y (any): input value
            model (str): which model used
            lamba (float): penalty parameter
            mindegree (int): minimum degree
            maxdegree (int): maximum degree
            k (int): number of folds
            max_iter (int): max iterations for Lasso
            seed (int): randomizer seed 

        Returns:
            mse (list): mean squared error per polynomial degree

    """
    kFold = KFold(n_splits=k, shuffle=True, random_state=seed)
    mse = []

    x = x.reshape(-1, 1)     #reshape x into a 2 dim column vector
    
    for deg in range(mindegree, maxdegree):
        if model == 'OLS':
            regression = LinearRegression()
        elif model == 'Ridge':
            regression = Ridge(alpha=lamba)
        elif model == 'Lasso':
            regression = Lasso(alpha=lamba, max_iter=max_iter)

        pipe = make_pipeline(PolynomialFeatures(degree=deg),
                             StandardScaler(),
                             regression)
         
        scores = -cross_val_score(pipe, x, y, cv=kFold,
                                     scoring='neg_mean_squared_error')
        mse.append(np.mean(scores))

    return mse
