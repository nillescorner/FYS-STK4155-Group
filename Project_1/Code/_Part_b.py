from general_functions import make_data, np
from regression_methods import regression
from plots import plot_regression

######## PART B ##########
"""Analysis for different values of penalty parameter lambda. Thus fixed sigma = 0.1 and fixed n = 100 (i.e default make data)"""

mindegree, maxdegree = 1, 16
x_axis_15 = np.arange(mindegree, maxdegree)
x,y = make_data()

lambas = [0.01, 0.5, 1.0]

mse_train_R, mse_test_R, = [], []
r2_train_R, r2_test_R = [], [] 
theta_norm_R = []

for i in range(len(lambas)):
    mse_train, mse_test, r2_train, r2_test, theta = regression(x, y, lamba=lambas[i])

    mse_train_R.append(mse_train);  mse_test_R.append(mse_test)
    r2_train_R.append(r2_train);    r2_test_R.append(r2_test)

    theta_norm_R.append([np.linalg.norm(theta) for theta in theta])


plot_regression(x_axis_15, mse_train_R, mse_test_R, titles=r'Mean squared error for Ridge with $\lambda$ = {}', params=lambas, ylabel='MSE', yscale='log')
plot_regression(x_axis_15, r2_train_R, r2_test_R, titles=r'R2 score for $\lambda$ = {}', params=lambas, ylabel='R2 score', yscale='linear')
plot_regression(x_axis_15, theta_norm_R, titles=r'$\theta$ for $\lambda$ = {}', params=lambas, ylabel=r'$\theta$', yscale='linear')


"""Analysis of Ridge regression dependence of the number of on datapoints n"""
ns = [100, 400, 4000]

lamba = 0.1
mse_train_n, mse_test_n = [], []
r2_train_n, r2_test_n = [], []
theta_n = []

for n in ns:
    x_n, y_n = make_data(n=n)
    mse_train, mse_test, r2_train, r2_test, thetas = regression(x_n, y_n, lamba=lamba)

    mse_train_n.append(mse_train);   mse_test_n.append(mse_test)
    r2_train_n.append(r2_train);    r2_test_n.append(r2_test)

    theta_n.append([np.linalg.norm(theta) for theta in thetas])

plot_regression(x_axis_15, mse_train_n, mse_test_n, titles='Mean squared error for OLS with n = {}', params=ns, ylabel='MSE', yscale='log')
plot_regression(x_axis_15, r2_train_n, r2_test_n, titles='R2 score for n = {}', params=ns, ylabel='R2 score', yscale='linear')
plot_regression(x_axis_15, theta_n, titles=r'$\theta$ for n = {}', params=ns, ylabel='R2 score', yscale='linear')


"""Analysis of Ridge regression dependence on noise (sigma). Thus fixed n = 100"""
sigmas = [0.05, 0.25, 0.75]

mse_train_s, mse_test_s = [], []
r2_train_s, r2_test_s = [], []
theta_norm_s = []

for sigma in sigmas:
    x_s, y_s = make_data(noise=sigma)
    mse_train, mse_test, r2_train, r2_test, thetas = regression(x_s,y_s, lamba=lamba)

    mse_train_s.append(mse_train);  mse_test_s.append(mse_test)
    r2_train_s.append(r2_train);    r2_test_s.append(r2_test)

    theta_norm_s.append([np.linalg.norm(theta) for theta in thetas])

plot_regression(x_axis_15, mse_train_s, mse_test_s, titles=r'Mean squared error for OLS with $\sigma$ = {}', params=ns, ylabel='MSE', yscale='log')
plot_regression(x_axis_15, r2_train_s, r2_test_s, titles=r'R2 score for $\sigma$ = {}', params=ns, ylabel='R2 score', yscale='linear')
plot_regression(x_axis_15, theta_norm_s, titles=r'$\theta$ for $\sigma$ = {}', params=ns, ylabel='R2 score', yscale='linear')
