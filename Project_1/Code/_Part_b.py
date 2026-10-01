""""
This python file contains the code used to derive the figures for mean squared error, r2 score
and theta as a functin of the polynomial degree for linear regression method: Ridge
This has been done for different penalty parameters (lambda), different data points n and different noise (sigma).

x and y arrays were generated using make_data function from general_functions.py, 
MSE, R2 score and Theta were derived using regression function from regression_methods.py,
The plots were generated using plot_regression function from plots.py
"""

from general_functions import make_data, np
from regression_methods import regression
from plots import plt, plot_heatmap_grid

"""Analysis for different values of penalty parameter lambda. Thus default data and fixed degree"""

x,y = make_data()   #default arrays
degree = 8
mse_train_R, mse_test_R, = [], []
r2_train_R, r2_test_R = [], [] 
theta_per_degree = []

lambas = np.logspace(-8,4, 1000)
for lamba in lambas:
    mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y, lamba, mindeg=degree, maxdeg=degree+1)
    mse_train_R.append(mse_train);  mse_test_R.append(mse_test)
    r2_train_R.append(r2_train);    r2_test_R.append(r2_test)

    theta_per_degree.append([np.linalg.norm(theta) for theta in thetas])

best_index = np.argmin(mse_test_R)
best_lambda = lambas[best_index]

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6.4,7))
fig.suptitle(fr'Analysis on how different $\lambda$s affect degree {degree} polynomial')
ax[0].plot(lambas, mse_train_R, color="#DD55FF", label='Training data')
ax[0].plot(lambas, mse_test_R, color="#0AC23E", label='Test data')
ax[0].axvline(x=best_lambda, color="#696968", ls='--', label=rf'Best $\lambda$ = {best_lambda:.4f}')
ax[0].legend()
ax[0].set_ylabel('Mean Squared error')

ax[1].plot(lambas, r2_train_R, color="#DD55FF", label='Training data')
ax[1].plot(lambas, r2_test_R, color="#0AC23E", label='Test data')
ax[1].axvline(x=best_lambda, color="#696968", ls='--', label=rf'Best $\lambda$ = {best_lambda}')
ax[1].set_ylabel('R2 score')

ax[2].plot(lambas, theta_per_degree, color="#9A2E2D")
ax[2].set_ylabel(r'$\|\theta\|_2$')
ax[2].set_yscale('log')
ax[2].set_xscale('log')
ax[2].set_xlabel(r'$\lambda$')
plt.show()

#For further analysis that depends on other parameters then lambda, we use the best lambda derived above.

"""Analysis of how the MSE, R2 score and Theta are affected by polynomial degree"""
mindegree, maxdegree = 1, 16
degrees = np.arange(mindegree, maxdegree)

mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y,lamba=best_lambda, mindeg=mindegree, maxdeg=maxdegree)
theta_per_degree = [np.linalg.norm(theta) for theta in thetas]

best_index = np.argmin(mse_test)
best_degree = degrees[best_index]

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6.4,7))
fig.suptitle(fr'Parameters dependence on Polynomial degree for Ridge with $\lambda$ = {best_lambda:.4f}')
ax[0].plot(degrees, mse_train, color="#DD55FF", label='Training data')
ax[0].plot(degrees, mse_test, color="#0AC23E", label='Test data')
ax[0].axvline(x=best_degree, color="#696968", ls='--', label=rf'Best degree = {best_degree}')
ax[0].legend()
ax[0].set_ylabel('Mean Squared error')

ax[1].plot(degrees, r2_train, color="#DD55FF", label='Training data')
ax[1].plot(degrees, r2_test, color="#0AC23E", label='Test data')
ax[1].axvline(x=best_degree, color="#696968", ls='--', label=rf'Best degree = {best_lambda}')
ax[1].set_ylabel('R2 score')

ax[2].plot(degrees, theta_per_degree, color="#9A2E2D")
ax[2].set_ylabel(r'$\|\theta\|_2$')
ax[2].set_yscale('log')
ax[2].set_xlabel('Polynomial Degree')
ax[2].set_xticks(degrees)
plt.show()

"""Analysis of how different amounts of data points (n) and different noise (sigma) affect affect
MSE and R2 score for the test data at the best degree"""
ns = np.linspace(40,4000, 50, dtype=int)
sigmas = np.linspace(0.01, 1.0, 50)

mse_test = np.zeros((len(sigmas), len(ns)))
r2_test = np.zeros_like(mse_test)

for i, sigma in enumerate(sigmas):
    for j, n in enumerate(ns):
        x, y = make_data(n, noise=sigma)
        _, mse_te, _, r2_te, theta = regression(x,y, lamba=best_lambda, mindeg=best_degree, maxdeg=best_degree+1)

        mse_test[i,j] = mse_te[0]
        r2_test[i,j] = r2_te[0]        

plot_heatmap_grid(mse_test, ns, sigmas, rf'Test MSE with Ridge at degree {best_degree} for $\lambda$ = {best_lambda:.4f}', 'MSE', log=True)
plot_heatmap_grid(r2_test, ns, sigmas, rf'Test R2 Score with Ridge at degree {best_degree} for $\lambda$ = {best_lambda:.4f}', 'R2', log=False)
plt.show()
