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
sigma = 0.1
x,y = make_data(n=100, noise=sigma)   #default data

degree = 15
mse_train_R, mse_test_R, = [], []
r2_train_R, r2_test_R = [], [] 
thetas_R = []

#Needed to calculate the degree of freedom, set intercept = False to avoid 0 column
X = design_matrix(x, degree=degree, intercept=False)
X_train, X_test, y_train, y_test = scaling(X, y, seed=2026, split_data=True)
s = np.linalg.svd(X_train, compute_uv=False)
dfs = []

#Based on w35tuesday.ipynb to calculate degrees of freedom
lambas = np.logspace(-8,4, 61)
for lamba in lambas:
    mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y, lamba, mindeg=degree, maxdeg=degree+1)
    mse_train_R.append(mse_train[0]);  mse_test_R.append(mse_test[0])
    r2_train_R.append(r2_train[0]);    r2_test_R.append(r2_test[0])
    thetas_R.append(thetas[0])
    dfs.append(np.sum(s**2 / (s**2 + lamba)))

theta_norm_R = [np.linalg.norm(theta) for theta in thetas_R]

#Find best degree given lambda
best_index = int(np.argmin(mse_test_R))
best_lambda = lambas[best_index]
best_mse = mse_test_R[best_index]
print('Polynomial as a function of lambda')
print(f'Best Test MSE Ridge as function of lambda {best_mse:.3e}')
print(f'Best Test R2 Ridge as function of lambda {r2_test_R[np.argmax(r2_test_R)]:.4f}', '\n')


fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(7,7))
fig.suptitle(fr'Analysis on how different $\lambda$s affect degree {degree} polynomial')
ax[0].plot(lambas, mse_train_R, color="#8C564B", label='Training data')
ax[0].plot(lambas, mse_test_R, color="#0AC23E", label='Test data')
ax[0].axvline(x=best_lambda, color="grey", ls='--', label=rf'Best $\lambda$ = {best_lambda:.3f}, df = {dfs[best_index]:.2f}')
ax[0].axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')
ax[0].legend(loc='lower right', bbox_to_anchor=(1.1, -0.15))
ax[0].set_ylabel('MSE')
ax[0].set_yscale('log')

ax[1].plot(lambas, r2_train_R, color="#8C564B")
ax[1].plot(lambas, r2_test_R, color="#0AC23E")
ax[1].axvline(x=best_lambda, color="grey", ls='--')

ax[2].plot(lambas, theta_norm_R, color="#001E43", label=r'$\|\theta\|_2$')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_ylabel(r'$\|\theta\|_2$')
ax[2].set_xlabel(r'$\lambda$')
ax[2].legend()
#plt.savefig(FIG_DIR / "Part_b_Params_vs_lambda.pdf")
plt.show()

plot_theta(thetas_R, lambas, xlabel=r'$\lambda$', title=rf'Ridge coefficients $\theta_j$ over $\lambda$ at degree {degree}', xlog=True, log=False)
#plt.savefig(FIG_DIR / "Part_b_theta_vs_lambda.pdf")
plt.show()

#For further analysis that depends on other parameters then lambda, we use the best lambda derived above.
"""Analysis of how the MSE, R2 score and Theta are affected by polynomial degree"""
mindegree, maxdegree = 1, 16
degrees = np.arange(mindegree, maxdegree)

mse_train, mse_test, r2_train, r2_test, thetas = regression(x,y,lamba=best_lambda, mindeg=mindegree, maxdeg=maxdegree)

best_index = np.argmin(mse_test)
best_degree = degrees[best_index]
best_mse = mse_test[best_index]
print('Polynomial as a function of degree')
print(f'Best MSE Ridge as function of polynomial {best_mse:.3e}')
print(f'Best R2 Score as a function of polynomial {r2_test[np.argmax(r2_test)]:.4f}')


theta_norms_R = [np.linalg.norm(theta) for theta in thetas]

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(7,7))
fig.suptitle(fr'Parameters dependence on Polynomial degree for Ridge with $\lambda$ = {best_lambda:.3f}')
ax[0].plot(degrees, mse_train, color='#8C564B', label='Training data')
ax[0].plot(degrees, mse_test, color='#0AC23E', label='Test data')
ax[0].axvline(x=best_degree, color='grey', ls='--', label=rf'Best degree = {best_degree}')
ax[0].axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')
ax[0].legend(loc='upper right', bbox_to_anchor=(1.05, 1.15))
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
#plt.savefig(FIG_DIR / "Part_b_Parms_vs_degree.pdf")
plt.show()

plot_theta(thetas, degrees, title=r'Coefficients of $\theta_j$ for Ridge over polynomial degree')
#plt.savefig(FIG_DIR / "Part_b_theta_vs_degree.pdf")
plt.show()


"""Analysis of how different amounts of data points (n) and different noise (sigma) affect affect
MSE and R2 score for the test data at the best degree"""
ns = np.linspace(40,4000, 15, dtype=int)
sigmas = np.linspace(0.01, 1.0, 10)

mse_test = np.zeros((len(sigmas), len(ns)))
r2_test = np.zeros_like(mse_test)

for i, sigma in enumerate(sigmas):
    for j, n in enumerate(ns):
        x, y = make_data(n, noise=sigma)
        _, mse_te, _, r2_te, theta = regression(x,y, lamba=best_lambda, mindeg=best_degree, maxdeg=best_degree+1)

        mse_test[i,j] = mse_te[0]
        r2_test[i,j] = r2_te[0]        

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

print('Heatmap analysis')
i,j = np.argwhere(mse_test==mse_test.min())[0]
print(f'Best test MSE = {mse_test[i,j]:.3e} at n = {ns[j]}, sigma = {sigmas[i]:.3e}')

i,j = np.argwhere(r2_test==r2_test.max())[0]
print(f'Best test R2 score  = {r2_test[i,j]:.4f} at n = {ns[j]}, sigma = {sigmas[i]:.3e}')


plot_heatmap_grid(mse_test, ns, sigmas, 'Test MSE', 'MSE', log=True, ax=axes[0], best='min', annotate=True)
plot_heatmap_grid(r2_test, ns, sigmas, r'Test $R^2$', r'$R^2$', ax=axes[1], best='max', annotate=True)
axes[0].set_xlabel('')  #they share x axis so xlabel on top plot is empty

fig.suptitle(rf'Dependence on data points and noise at degree {best_degree} for Ridge with $\lambda$ = {best_lambda:.3f}')
fig.tight_layout()
#plt.savefig(FIG_DIR / "Part_b_heatmaps_MSE_R2.pdf")
plt.show()


