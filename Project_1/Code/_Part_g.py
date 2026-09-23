from lasso_methods import *
from general_functions import *

degree = 5
lam = 1e-2
n = 100

#making data
x_raw, y_raw = make_data(n = n)
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, y = scaling(X_raw, y_raw, split_data = False)
theta0 = np.zeros(X.shape[1])

gamma_lasso = lam * n / 2.0

"""Gradient descent"""
history_lasso_gd, n_lasso_gd = lasso_gd(X, y, gamma_lasso, lmbda = lam)

"""Coordinate descent"""
history_lasso_cd, n_lasso_cd = lasso_coordinate_descent(X, y, lmbda=lam)
