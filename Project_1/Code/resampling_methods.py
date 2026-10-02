"""
This python file contains the functions necessary to do the resampling techniques bootstrap and cross-validation
"""

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.base import clone

def bootstrap_resampling(x,y, mindegree=1, maxdegree=21, n_bootstraps=100, seed=2026, best_estimator = None, test_wholedataset=False):
    """

    Bootstrap resampling function for simpler ordinary least squares based on code from p.65, or predefined estimator given in 'estimator'

        Params:
            x (any): input values.
            y (any): input values. 
            mindegree (int): minimum degree
            maxdegree (int): maximum degree
            n_bootstraps (int): number of iterations for bootstrap
            seed (int): randomizer seed
            best_estimator(scikit learn estimator) : predefined estimator
            test_wholedataset (bool): if True, resampling uses the whole dataset

        Returns:
            error (list): test error for each polynomial degree
            bias (list): bias for each polynomial degree
            variance (list): variance for each polynomial degree
    """      
                       
    x = x.reshape(-1,1)

    if test_wholedataset:
        x_train, x_test, y_train, y_test = x, x, y, y
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)


    error, bias, variance = [], [], []

    for deg in range(mindegree,maxdegree):
        if best_estimator == None:
            #StandardScaler does X_norm, applies linear regression 
            model = make_pipeline(PolynomialFeatures(degree=deg),
                                StandardScaler(),
                                LinearRegression())
        else:
            model = estimator = make_pipeline(
                                PolynomialFeatures(degree=deg),
                                clone(best_estimator),
                            )
        
        

        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train, random_state= i+1)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

        error.append(np.mean(np.mean((y_test[:, None] - y_pred)**2, axis=1)))
        bias.append(np.mean((y_test[:, None] - np.mean(y_pred, axis=1, keepdims=True))**2))
        variance.append(np.mean(np.var(y_pred, axis=1)))

    return error, bias, variance


def kfold_resampling(x,y,k, best_est, mindegree=1, maxdegree=21, n_resamples=100, seed=2026, test_wholedataset = False):
    """

    kfold resampling function for simpler ordinary least squares based on code from p.65, or predefined estimator given in 'estimator'

        Params:
            x (any): input values.
            y (any): input values. 
            k (int) : k-folds
            mindegree (int): minimum degree
            maxdegree (int): maximum degree
            n_resamples (int): number of iterations for resamples
            seed (int): randomizer seed
            test_wholedataset (bool): if True, resampling uses the whole dataset

        Returns:
            error (list): test error for each polynomial degree
            bias (list): bias for each polynomial degree
            variance (list): variance for each polynomial degree
    """      

    if test_wholedataset:
        pass
    else:                    
        x = x.reshape(-1,1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        x = x_train
        y = y_train


    error, bias, variance = [], [], []

    for deg in range(mindegree,maxdegree):
        X = PolynomialFeatures(degree=deg).fit_transform(x.reshape(-1, 1))

        y_pred = np.empty((y.shape[0], n_resamples))
        for r in range(n_resamples):
            kf = KFold(k, shuffle=True, random_state= seed + r)
            for tr, va in kf.split(X):
                model = clone(best_est).fit(X[tr], y[tr])   # fixed alpha, no re-tuning
                y_pred[va,r] = model.predict(X[va])

        error.append(np.mean(np.mean((y[:, None] - y_pred)**2, axis=1)))
        bias.append(np.mean((y[:, None] - np.mean(y_pred, axis=1, keepdims=True))**2))
        variance.append(np.mean(np.var(y_pred, axis=1)))

    return error, bias, variance



def cross_validation(x, y, model='OLS', lamba=0.0, mindegree=1, maxdegree=21, k=5, max_iter=10000, seed=2026):
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


def grid_search(x_train, x_test, y_train, k,  param_grid = {"ridge__alpha": np.logspace(-5, 3, 30)},
                  model='Ridge', lamba=0.0,
                  mindegree=1, maxdegree=21, max_iter=10000,  seed=2026,
                  return_X = False):

    """
    Uses cross-validation to find the best regularization parameter for Ridge
    or Lasso models at each polynomial degree.

    Params:
        x_train (array): Training input values.
        x_test (array): Test input values, transformed for each polynomial degree.
        y_train (array): Training target values.
        k (int): Number of cross-validation folds.
        param_grid (dict): Parameter values to test in GridSearchCV. The key
            should match the selected model, e.g. "ridge__alpha" or "lasso__alpha".
        model (str): Model to use; accepts "Ridge", "ridge", "Lasso", or "lasso".
        lamba (float): Initial regularization parameter passed to the estimator.
        mindegree (int): Minimum polynomial degree to evaluate.
        maxdegree (int): Upper bound for polynomial degrees (not included).
        max_iter (int): Maximum iterations for Lasso.
        seed (int): Randomizer seed for cross-validation fold shuffling.
        return_X (bool): If True, also returns fitted models and transformed
            training and test inputs.

    Returns:
        gridsearch (dict): GridSearchCV objects indexed by polynomial degree.
            If return_X is True, returns a tuple containing gridsearch, fitted
            models, and transformed training and test inputs.
    """

    gridsearch = {}
    fit = {}
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

        fit[deg] = gridsearch[deg].fit(X_train_dict[deg], y_train)
    if return_X:
        return gridsearch, fit, X_train_dict, X_test_dict
    else:
        return gridsearch




def train_through_gridsearchCV(x,y, models = ['ridge'], Ks=[5,10],   mindegree=1, maxdegree=21, lambas = np.logspace(-6,3,30), seed = 2026):
    """
    Splits the data into training and test sets, then runs Ridge and Lasso
    grid searches for each fold count and polynomial degree.

    Params:
        x (array): Input values.
        y (array): Target values.
        models (list) : contains which models to gridsearch for. Either 'ridge' or 'lasso' or both
        Ks (list): Numbers of cross-validation folds to evaluate.
        mindegree (int): Minimum polynomial degree to evaluate.
        maxdegree (int): Upper bound for polynomial degrees (not included).
        lambas (array): Candidate regularization parameter values.
        seed (int): Randomizer seed for the train-test split and cross-validation.

    Returns:
        results (dict): Results indexed by model and fold count. Each entry
            contains the grid searches and fitted models indexed by degree,
            best regularization parameters, cross-validation MSE, and test MSE.
        model_shorthands (list): Keys used to index results, such as
            "ridge, folds = 5".
    """
    degs = np.arange(mindegree, maxdegree)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=seed)

    results = {} #for storing our results
    model_shorthands = [] #easy way to retrieve model data without recalculating everyhting
 
    for k in Ks:
        for model in models:
            param = {f"{model}__alpha" : lambas}    
            modelsearch, fit,  X_train_dict, X_test_dict = grid_search(x_train, x_test, y_train, k, model = model, param_grid = param, return_X = True,
                                                                        maxdegree = maxdegree, mindegree=mindegree)
        
            best_params = []
            mse = []
            score = []
            for deg in degs:
                best_params.append(modelsearch[deg].best_params_[f"{model}__alpha"])
                mse.append(- modelsearch[deg].best_score_)
                score.append(- modelsearch[deg].score(X_test_dict[deg], y_test)) #NOTE: need to figure out how .scoring works

            model_shorthands.append(f"{model} folds = {k}")
            results[f"{model} folds = {k}"] = {'model': modelsearch, #indexed by degree
                                                'fit' : fit, #indexed by degree
                                                'best_params': best_params,
                                                'mse' : mse,
                                                'score': score} 
        
    return results, model_shorthands

def mse_decomposer(x, y, results, model_shorthands, resamples = 100,  method ='kfold_resampling', mindegree=1, maxdegree=21):
    """
    Estimates prediction error, squared bias, and variance for each model and
    polynomial degree using resampling.

    Params:
        x (array): Input values to evaluate.
        y (array): Target values corresponding to x.
        results (dict): Results returned by train_through_gridsearchCV.
        model_shorthands (list): Model and fold-count keys from results.
        resamples (int): Number of resampling runs.
        method (str): Resampling method; use "kfold_resampling" or
            "bootstrap_resampling".
        mindegree (int): Minimum polynomial degree to evaluate.
        maxdegree (int): Upper bound for polynomial degrees (not included).

    Returns:
        mse_decomposition (dict): Results indexed by model and fold count.
            Each entry contains arrays of error, squared bias plus noise, and
            variance, ordered by polynomial degree. 

    Raises:
        ValueError: If method is not "kfold_resampling" or
            "bootstrap_resampling".

    """

    if method not in ('kfold_resampling', 'bootstrap_resampling'):
        raise ValueError(
            "method must be 'kfold_resampling' or 'bootstrap_resampling'"
        )

    mse_decomposition = {}
    degs = np.arange(mindegree, maxdegree)


    for model_shorthand in model_shorthands:
        model_name, _, _, k = model_shorthand.split()
        k = int(k)

        mse_decomposition[model_shorthand] = {}
        error, bias2_plus_noise, variance = (np.zeros(maxdegree-mindegree) for _ in range(3))
        for idx, deg in enumerate(degs):
            deg = int(deg)
            
            best_estimator = results[model_shorthand]['fit'][deg].best_estimator_

            if method == 'kfold_resampling':
                error_, bias2_plus_noise_, variance_ = kfold_resampling(x, y, k, best_estimator, deg, deg+1, n_resamples = resamples, test_wholedataset  = True) #uses whole dataset for fitting model
            elif method =='bootstrap_resampling':
                error_, bias2_plus_noise_, variance_ = bootstrap_resampling(x, y, deg, deg+1, best_estimator = best_estimator, n_bootstraps= resamples, test_wholedataset  = True)

            error[idx], bias2_plus_noise[idx], variance[idx] = error_[0], bias2_plus_noise_[0], variance_[0] 
        mse_decomposition[model_shorthand] = {'error': error,
                                                    'bias2_plus_noise': bias2_plus_noise,
                                                    'variance': variance }
    return mse_decomposition

# print(f"Best parameters: {search.best_params_}")
# print(f"Best cross-validation score (MSE): {search.best_score_:.4f}")
# print(f"Test set score: {search.score(X_test, y_test):.4f}")
