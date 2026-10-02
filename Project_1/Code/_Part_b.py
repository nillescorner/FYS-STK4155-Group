""""
This python file contains the code used to derive the figures for mean squared error, r2 score
and theta as a functin of the polynomial degree for linear regression method: Ridge
This has been done for different penalty parameters (lambda), different data points n and different noise (sigma).

x and y arrays were generated using make_data function from general_functions.py, 
MSE, R2 score and Theta were derived using regression function from regression_methods.py,
The plots were generated using plot_heatmap_grid and plot_theta functions from plots.py
"""

from general_functions import make_data, np, design_matrix, scaling, FIG_DIR
from regression_methods import regression
from plots import plt, plot_heatmap_grid, plot_theta

"""Analysis for different values of penalty parameter lambda. Let's Ridge's lambda do the model selection"""
x,y = make_data()   #default arrays
degree = 15
mse_train_R, mse_test_R, = [], []
r2_train_R, r2_test_R = [], [] 
thetas_R = []

#Needed to calculate the degree of freedom, intercept = False to avoid 
X = design_matrix(x, degree=degree, intercept=False)
X_train, X_test, y_train, y_test = scaling(X, y, seed=2026, split_data=True)
s = np.linalg.svd(X_train, compute_uv=False)
dfs = []

lambas = np.logspace(-8,4, 61)
for lamba in lambas:
    mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y, lamba, mindeg=degree, maxdeg=degree+1)
    mse_train_R.append(mse_train[0]);  mse_test_R.append(mse_test[0])
    r2_train_R.append(r2_train[0]);    r2_test_R.append(r2_test[0])
    thetas_R.append(thetas[0])
    dfs.append(np.sum(s**2 / (s**2 + lamba)))

theta_norm_R = [np.linalg.norm(theta) for theta in thetas_R]

best_index = int(np.argmin(mse_test_R))
best_lambda = lambas[best_index]

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(7,7))
fig.suptitle(fr'Analysis on how different $\lambda$s affect degree {degree} polynomial')
ax[0].plot(lambas, mse_train_R, color="#DD55FF", label='Training data')
ax[0].plot(lambas, mse_test_R, color="#0AC23E", label='Test data')
ax[0].axvline(x=best_lambda, color="#696968", ls='--', label=rf'Best $\lambda$ = {best_lambda:.3f}, df = {dfs[best_index]:.2f}')
ax[0].legend()
ax[0].set_ylabel('MSE')
ax[0].set_yscale('log')

ax[1].plot(lambas, r2_train_R, color="#DD55FF")
ax[1].plot(lambas, r2_test_R, color="#0AC23E")
ax[1].axvline(x=best_lambda, color="#696968", ls='--')

ax[2].plot(lambas, theta_norm_R, color="#001E43", label=r'$\|\theta\|_2$')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_ylabel(r'$\|\theta\|_2$')
ax[2].legend()
plt.show()

plot_theta(thetas_R, lambas, xlabel=r'$\lambda$', title='Coefficients for $\\theta_j$ over $\\lambda$ at degree 15', xlog=True, log=False)
plt.show()

"""Shrinkage, through the SVD"""
sigs = np.logspace(np.log10(s.min())- 1.0, np.log10(s.max()) + 1.0, 100)
lambas_shrink = [1e-8, 1e-6, 1e-4, 1e-2, 1]

fig, ax = plt.subplots()
for lam in lambas_shrink:
    ax.plot(sigs, sigs**2 / (sigs**2 + lam), label=rf'$\lambda$ = {lam}')

ax.set_xscale('log')
ax.set_title('Shrinkage through SVD')
ax.set_xlabel(r'Singular value $\sigma_i$')
ax.set_ylabel(r'Shrinkage $\sigma_i^2$/($\sigma_i^2 + \lambda$)')
ax.legend()
plt.show()

#For further analysis that depends on other parameters then lambda, we use the best lambda derived above.
"""Analysis of how the MSE, R2 score and Theta are affected by polynomial degree"""
mindegree, maxdegree = 1, 16
degrees = np.arange(mindegree, maxdegree)

mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y,lamba=best_lambda, mindeg=mindegree, maxdeg=maxdegree)

best_index = np.argmin(mse_test)
best_degree = degrees[best_index]
theta_norms_R = [np.linalg.norm(theta) for theta in thetas]

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(7,7))
fig.suptitle(fr'Parameters dependence on Polynomial degree for Ridge with $\lambda$ = {best_lambda:.3f}')
ax[0].plot(degrees, mse_train, color='#8C564B', label='Training data')
ax[0].plot(degrees, mse_test, color='#0AC23E', label='Test data')
ax[0].axvline(x=best_degree, color='#696968', ls='--', label=rf'Best degree = {best_degree}')
ax[0].legend()
ax[0].set_yscale('log')
ax[0].set_ylabel('MSE')

ax[1].plot(degrees, r2_train, color='#8C564B')
ax[1].plot(degrees, r2_test, color='#0AC23E')
ax[1].axvline(x=best_degree, color='grey', ls='--')
ax[1].set_ylabel('R2')

ax[2].plot(degrees, theta_norms_R, color="#001E43", label=r'$\|\theta\|_2$')
ax[2].set_yscale('log')
ax[2].set_ylabel(r'$\|\theta\|_2$')
ax[2].set_xlabel('Polynomial Degree')
ax[2].set_xticks(degrees)
ax[2].legend()
plt.show()

plot_theta(thetas, degrees, title=r'Coefficients of $\theta_j$ for Ridge over polynomial degree')
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

plot_heatmap_grid(mse_test, ns, sigmas, rf'Test MSE with Ridge at degree {best_degree} for $\lambda$ = {best_lambda:.3f}', 'MSE', log=True)
plot_heatmap_grid(r2_test, ns, sigmas, rf'Test R2 Score with Ridge at degree {best_degree} for $\lambda$ = {best_lambda:.3f}', 'R2', log=False)
#plt.show()


