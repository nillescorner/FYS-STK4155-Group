"""
This python file contains the code used to derive the figures for mean squared error, r2 score
and theta as a functin of the polynomial degree for linear regression method: Ordinary Least Squares (OLS)
This has been done for different data points n and different noise (sigma).

x and y arrays were generated using make_data function from general_functions.py, 
MSE, R2 score and Theta were derived using regression function from regression_methods.py,
The plots were generated using plot_regression function from plots.py
"""

from general_functions import make_data, np
from regression_methods import regression
from plots import plot_regression


"""Analysis of OLS regression dependence of the number of on datapoints n"""
ns = [100, 400, 4000]

mindegree, maxdegree = 1, 16
x_axis_15 = np.arange(mindegree, maxdegree)

mse_train_n, mse_test_n = [], []
r2_train_n, r2_test_n = [], []
theta_n = []

for n in ns:
    x_n, y_n = make_data(n=n)   #generate arrays for different number of data points
    mse_train_ols, mse_test_ols, r2_train_ols, r2_test_ols, thetas_ols = regression(x_n, y_n)

    mse_train_n.append(mse_train_ols);  mse_test_n.append(mse_test_ols)
    r2_train_n.append(r2_train_ols);    r2_test_n.append(r2_test_ols)

    theta_n.append([np.linalg.norm(theta) for theta in thetas_ols])

plot_regression(x_axis_15, mse_train_n, mse_test_n, titles='Mean squared error for OLS with n = {}', params=ns, ylabel='MSE', yscale='log')
plot_regression(x_axis_15, r2_train_n, r2_test_n, titles='R2 score for n = {}', params=ns, ylabel='R2 score', yscale='linear')
plot_regression(x_axis_15, theta_n, titles=r'$\theta$ for n = {}', params=ns, ylabel='R2 score', yscale='linear')


"""Analysis of OLS regression dependence on noise (sigma). Thus fixed n = 100"""
sigmas = [0.05, 0.25, 0.75]

mse_train_s, mse_test_s = [], []
r2_train_s, r2_test_s = [], []
theta_norm_s = []

for sigma in sigmas:
    x_s, y_s = make_data(noise=sigma)   #generate arrays for different noise values
    mse_train_ols, mse_test_ols, r2_train_ols, r2_test_ols, thetas_ols = regression(x_s,y_s)

    mse_train_s.append(mse_train_ols);  mse_test_s.append(mse_test_ols)
    r2_train_s.append(r2_train_ols);r2_test_s.append(r2_test_ols)

    theta_norm_s.append([np.linalg.norm(theta) for theta in thetas_ols])

plot_regression(x_axis_15, mse_train_s, mse_test_s, titles=r'Mean squared error for OLS with $\sigma$ = {}', params=ns, ylabel='MSE', yscale='log')
plot_regression(x_axis_15, r2_train_s, r2_test_s, titles=r'R2 score for $\sigma$ = {}', params=ns, ylabel='R2 score', yscale='linear')
plot_regression(x_axis_15, theta_norm_s, titles=r'$\theta$ for $\sigma$ = {}', params=ns, ylabel='R2 score', yscale='linear')
