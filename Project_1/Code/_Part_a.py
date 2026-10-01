"""
This python file contains the code used to derive the figures for mean squared error, r2 score
and theta as a functin of the polynomial degree for linear regression method: Ordinary Least Squares (OLS)
This has been done for different data points n and different noise (sigma).

x and y arrays were generated using make_data function from general_functions.py, 
MSE, R2 score and Theta were derived using regression function from regression_methods.py,
The plots were generated using plot_train_test and plot_heatmap_grid functions from plots.py
"""

from general_functions import make_data, np
from regression_methods import regression
from plots import plot_heatmap_grid, plt

"""Analysis of how polynomial degree affects MSE, R2 and Theta"""
x, y = make_data()  #default data
mindegree, maxdegree = 1, 16
degrees = np.arange(mindegree,maxdegree)
#Derive data needed to plot
mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y, mindeg=mindegree, maxdeg=maxdegree)
theta_per_degree = [np.linalg.norm(theta) for theta in thetas]

best_index = np.argmin(mse_test)
best_degree = degrees[best_index]

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6.4,7))
fig.suptitle('Parameters dependence on Polynomial degree for OLS')

ax[0].plot(degrees, mse_train, color="#DD55FF", label='Training data')
ax[0].plot(degrees, mse_test, color="#0AC23E", label='Test data')
ax[0].axvline(x=best_degree, color="#696968", ls='--', label=f'Best degree = {best_degree}')
ax[0].legend()
ax[0].set_ylabel('Mean Squared error')

ax[1].plot(degrees, r2_train, color='#DD55FF')
ax[1].plot(degrees, r2_test, color='#0AC23E')
ax[1].axvline(x=best_degree, color='#696968', ls='--')
ax[1].set_ylabel('R2 Score')

ax[2].plot(degrees, theta_per_degree, color="#9A2E2D")
ax[2].set_ylabel(r'$\|\theta\|_2$')
ax[2].set_xticks(degrees)
ax[2].set_yscale('log')
ax[2].set_xlabel('Polynomial Degree')
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
        _, mse_te, _, r2_te, theta = regression(x,y, mindeg=best_degree, maxdeg=best_degree+1)

        mse_test[i,j] = mse_te[0]
        r2_test[i,j] = r2_te[0]        

plot_heatmap_grid(mse_test, ns, sigmas, f'Test MSE using OLS at degree {best_degree}', 'MSE', log=True)
plot_heatmap_grid(r2_test, ns, sigmas, f'Test R2 Score using OLS at degree {best_degree}', 'R2', log=False)
plt.show()
