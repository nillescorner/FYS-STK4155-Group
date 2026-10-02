"""
This python file contains the code used to derive the figures for mean squared error, r2 score
and theta as a functin of the polynomial degree for linear regression method: Ordinary Least Squares (OLS)
This has been done for different data points n and different noise (sigma).

x and y arrays were generated using make_data function from general_functions.py, 
MSE, R2 score and Theta were derived using regression function from regression_methods.py,
The plots were generated using plot_heatmap_grid function from plots.py
"""

from general_functions import make_data, np, FIG_DIR
from regression_methods import regression
from plots import plot_heatmap_grid, plt, plot_theta

"""Analysis of how polynomial degree affects MSE, R2 and Theta"""
sigma = 0.1
x, y = make_data(n=100, noise=sigma)  #default data
mindegree, maxdegree = 1, 16
degrees = np.arange(mindegree,maxdegree)

mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y, mindeg=mindegree, maxdeg=maxdegree)
theta_norm_per_deg = [np.linalg.norm(theta) for theta in thetas]
best_index = np.argmin(mse_test)
best_degree = degrees[best_index]
best_mse = mse_test[best_index]

print(f'Best MSE for as function of polynomial = {best_mse:.4f}', '\n')

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(7,7))
fig.suptitle('Parameters dependence on Polynomial degree for OLS')
ax[0].plot(degrees, mse_train, color='#8C564B', label='Training data')
ax[0].plot(degrees, mse_test, color='#0AC23E', label='Test data')
ax[0].axvline(x=best_degree, color='grey', ls='--', label=f'Best degree = {best_degree}')
ax[0].axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')
ax[0].legend(loc='upper right', bbox_to_anchor=(1.05, 1.3))
ax[0].set_ylabel('MSE')
ax[0].set_yscale('log')

ax[1].plot(degrees, r2_train, color='#8C564B')
ax[1].plot(degrees, r2_test, color='#0AC23E')
ax[1].axvline(x=best_degree, color='#696968', ls='--')
ax[1].axhline(sigma**2, color='black', ls=':')
ax[1].set_ylabel('R2')

ax[2].plot(degrees, theta_norm_per_deg, color="#001E43", label=r'$\|\theta\|_2$')
ax[2].set_yscale('log')
ax[2].set_ylabel(r'$\|\theta\|_2$')
ax[2].set_xlabel('Polynomial Degree')
ax[2].legend()
ax[2].set_xticks(degrees)
plt.savefig(FIG_DIR / "Part_a_Paramets_over_polynomial.pdf")
plt.show()

plot_theta(thetas, degrees, title=r'Coefficients $\theta_j$ for OLS over polynomial degree')
plt.savefig(FIG_DIR / "Part_a_theta_coeff.pdf")
plt.show()

"""Analysis of how different amounts of data points (n) and different noise (sigma) affect
MSE and R2 score for the test data at the best degree"""
ns = np.linspace(40,4000, 50, dtype=int)
sigmas = np.linspace(0.01, 1.0, 50)

mse_test = np.zeros((len(sigmas), len(ns)))
r2_test = np.zeros_like(mse_test)

for i, sigma in enumerate(sigmas):
    for j, n in enumerate(ns):
        #Make dataset corresponding to this number of data points and noise
        x, y = make_data(n, noise=sigma)
        #Keep only test mse and test r2 score
        _, mse_te, _, r2_te, _ = regression(x,y, mindeg=best_degree, maxdeg=best_degree+1)

        #Fill corresponding element with this value given row = sigma, column = n
        mse_test[i,j] = mse_te[0]
        r2_test[i,j] = r2_te[0]        

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

plot_heatmap_grid(mse_test, ns, sigmas, 'Test MSE', 'MSE', log=True, ax=axes[0])
plot_heatmap_grid(r2_test, ns, sigmas, r'Test $R^2$', r'$R^2$', log=False, ax=axes[1])
axes[0].set_xlabel('')  #they share x axis so xlabel on top plot is empty

fig.suptitle(f'Dependence on data points and noise at degree {best_degree} for OLS')
fig.tight_layout()
plt.savefig(FIG_DIR / "Part_a_heatmaps_MSE_R2.pdf")
plt.show()
