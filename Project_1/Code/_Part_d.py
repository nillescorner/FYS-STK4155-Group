from general_functions import make_data, np
from resampling_methods import bootstrap_resampling, cross_validation
from regression_methods import regression
import matplotlib.pyplot as plt

mindegree, maxdegree = 1, 21
x_axis_20 = np.arange(mindegree, maxdegree)
x,y = make_data()

mse_cv_5_OLS = cross_validation(x,y, model='OLS',k=5)
mse_cv_10_OLS = cross_validation(x,y, model='OLS', k=10)
_, mse_test_ols_compare, _, _, _ = regression(x,y, maxdeg=maxdegree)
bootstrap_compare, *_ = bootstrap_resampling(x, y, maxdegree=maxdegree)

#Compare bootstrap, OLS and CV
fig, ax = plt.subplots(figsize=(8,4))
ax.set_title('Mean squared error comparison different methods')
ax.plot(x_axis_20, mse_test_ols_compare, 'o-',label='OLS MSE')
ax.plot(x_axis_20, bootstrap_compare, 'o-', label='Bootstrap')
ax.plot(x_axis_20, mse_cv_5_OLS, 's-', label='Cross value MSE kfold=5')
ax.plot(x_axis_20,mse_cv_10_OLS, 'd-', label='Cross value MSE kfold=10')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE')
ax.set_xticks(x_axis_20)
ax.set_yscale('log')
ax.legend()
plt.savefig(fname="figs/Part_d_MSE_comparison.pdf")
plt.show()


""""Analysis of Ridge Regression for Cross-validation method"""
n = 100
lambdas = np.logspace(-8,5,n)

degis = np.arange(mindegree, maxdegree)
mse_polynomials = np.zeros((len(lambdas), len(degis)))
spread_polynomials = np.zeros((len(lambdas), len(degis)))

for j, lmb in enumerate(lambdas):
    mse = cross_validation(x, y, model='Ridge', lamba=lmb, mindegree=mindegree, maxdegree=maxdegree)

    mse_polynomials[j, :] = mse #fills current lambda row and fills it with mse values for that row

fig, ax = plt.subplots(figsize=(8,5))
for d_idx, deg in enumerate(degis):
    b = np.argmin(mse_polynomials[:, d_idx])
    print(f"  degree = {deg:2d}:  best_lambda = {lambdas[b]:8.4g}"
          f"   CV-MSE = {mse_polynomials[b, d_idx]:.4f}")
    ax.plot(lambdas, mse_polynomials[:, d_idx], label=f'deg={deg}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('MSE (CV)')
ax.set_title('Ridge CV-MSE vs lambda for different polynomial degrees')
ax.legend(ncol=2, fontsize=8)
plt.savefig(fname="figs/Part_d_RidgeMSE_vsLambda.pdf")
plt.show()