"""Resampling techniques: bootstrap and cross-validation"""

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


def bootstrap_resampling(x,y, mindegree=1, maxdegree=21, n_bootstraps=100, seed=2026):
    """

    Bootstrap resampling function for simpler ordinary least squares based on p.65
    """                                
    x = x.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    error, bias, variance = [], [], []

    for deg in range(mindegree,maxdegree):
        #StandardScaler does X_norm, LinearRegression 
        model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False),
                              StandardScaler(),
                              LinearRegression())

        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train, random_state=seed + i)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

        error.append(np.mean(np.mean((y_test[:, None] - y_pred)**2, axis=1)))
        bias.append(np.mean((y_test[:, None] - np.mean(y_pred, axis=1, keepdims=True))**2))
        variance.append(np.mean(np.var(y_pred, axis=1)))

    return error, bias, variance


def cross_validation(x, y, model='ols', lamba=0.0, mindegree=1, maxdegree=21, k=5, max_iter=10000, seed=2026, ypred = False):
    """
    Cross-validation resampling technique. Works for OLS, Ridge, Lasso

    Returns MSE and y_pred
    """
    kFold = KFold(n_splits=k, shuffle=True, random_state=seed)
    mse = []

    x = x.reshape(-1, 1)     #reshape x into a 2 dim column vector
    y_pred = { }
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

        if ypred:
            #prediction of y, obtained by cross validation. use for visualisation purposes
            y_pred[deg] = cross_val_predict(pipe, x, y, cv=kFold)
    
    if ypred:
        return mse, y_pred
    else:
        return mse


def grid_search(x_train, x_test, y_train, k,  param_grid = {"ridge__alpha": np.logspace(-5, 3, 30)},
                  model='Ridge', lamba=0.0,
                  mindegree=1, maxdegree=21, max_iter=10000,  seed=2026,
                  return_X = False):

    """
    Grid search technique using cross validation. 
    Repeats process for all given degrees and returns a dict indexed by degrees containing GridSearchCV objects.

    Only includes Ridge and Lasso, as OLS doesn't have hyperparameters aside from amount of degrees
    
    """

    gridsearch = {}
    X_train_dict = {}
    X_test_dict = {}
    for deg in range(mindegree, maxdegree):
        if model == 'Ridge' or model == 'ridge':
            regression = Ridge(alpha=lamba)
        elif model == 'Lasso' or model == 'lasso':
            regression = Lasso(alpha=lamba, max_iter=max_iter)

        polyfit = PolynomialFeatures(degree=deg)
        X_train_dict[deg] = polyfit.fit_transform(x_train.reshape(-1, 1))
        X_test_dict[deg] = polyfit.fit_transform(x_test.reshape(-1, 1))


        pipe = make_pipeline(
            StandardScaler(),
            regression
        )

        gridsearch[deg] = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="neg_mean_squared_error", #MSE scoring 
            cv= KFold(n_splits=k, shuffle=True, random_state=seed), #sets cross-validation splitting strategy to the same as in cross_validation
            refit=True,  #Refit an estimator using the best found parameters on the whole dataset
            return_train_score = True, #returns training score 
        )

        gridsearch[deg].fit(X_train_dict[deg], y_train)
    if return_X:
        return gridsearch, X_train_dict, X_test_dict
    else:
        return gridsearch
