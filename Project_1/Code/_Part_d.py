""""
This python file contains the code used to derive the figure that shows MSE for OLS analysis
from parts a and c as a function of polynomial degree against the MSE from cross-validation for k = 5 and k = 10.
It also derives the plot for Ridge regression that shows different polynomial degrees are affected by penalty parameter lambda 

x and y arrays were generated using make_data function from general_functions.py, 
Test MSE for OLS was derived using regression function from regression_methods.py,
Test MSE from bootstrap resampling was derived using bootstrap_resampling function from resampling_methods.py
Test MSE and Ridge regression as a function of both polynomial degree and penalty parameter (lambda) was derived from cross_validation function from resampling.py
"""


from general_functions import make_data, np, FIG_DIR
from resampling_methods import bootstrap_resampling, cross_validation
from regression_methods import regression
from plots import plt

"""Analysis of the MSE as a function of polynomial degree for OLS and Bootstrap, Cross-validation"""
mindegree, maxdegree = 1, 21
degrees = np.arange(mindegree, maxdegree)
x,y = make_data()   #default data

mse_cv_5_OLS = cross_validation(x,y, model='OLS',k=5)
mse_cv_10_OLS = cross_validation(x,y, model='OLS', k=10)
_, mse_test_ols_compare, _, _, _ = regression(x,y, maxdeg=maxdegree)
bootstrap_compare, *_ = bootstrap_resampling(x, y, maxdegree=maxdegree)

#Compare bootstrap, OLS and CV
fig, ax = plt.subplots(figsize=(8,4))
ax.set_title('Mean squared error comparison different methods')
ax.plot(degrees, mse_test_ols_compare, 'o-', color='#F433DA',label='OLS MSE')
ax.plot(degrees, bootstrap_compare, 'o-',  color='#FF2B2B', label='Bootstrap')
ax.plot(degrees, mse_cv_5_OLS, 's-', color="#0F2D9A",label='Cross value MSE kfold=5')
ax.plot(degrees,mse_cv_10_OLS, 'd-', color="#FB7100", label='Cross value MSE kfold=10')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE')
ax.set_xticks(degrees)
ax.set_yscale('log')
ax.legend()
plt.savefig(fname="figs/Part_d_MSE_comparison.pdf")
plt.show()


""""
Analysis of Ridge Regression for Cross-validation method over different lambdas
Divided degrees into two for figure to be clearer and not so compact
"""
n = 100
lambdas = np.logspace(-8,4,n)
degrees = np.arange(1, 21)

mse_ridge = {}
mse_polynomials = np.zeros((len(lambdas), len(degrees)))
spread_polynomials = np.zeros((len(lambdas), len(degrees)))

for j, lmb in enumerate(lambdas):
    mse = cross_validation(x, y, model='Ridge', lamba=lmb, mindegree=mindegree, maxdegree=maxdegree)

    mse_polynomials[j, :] = mse #fills current lambda row and fills it with mse values for that row

fig, ax = plt.subplots(figsize=(8,5))
for index_degree, deg in enumerate(degrees):
    best = np.argmin(mse_polynomials[:, index_degree])
    print(f"  degree = {deg:2d}:  best_lambda = {lambdas[best]:8.4g}"
          f"   CV-MSE = {mse_polynomials[best, index_degree]:.4f}")
    ax.plot(lambdas, mse_polynomials[:, index_degree], label=f'deg={deg}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('MSE (CV)')
ax.set_title('Ridge CV-MSE vs lambda for different polynomial degrees')
ax.legend(ncol=2, fontsize=8)
plt.savefig(fname="figs/Part_d_RidgeMSE_vsLambda.pdf")
plt.show()