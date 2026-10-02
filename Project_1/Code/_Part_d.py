""""
This python file contains the code used to derive the figure that shows MSE for OLS analysis
from parts a and c as a function of polynomial degree against the MSE from cross-validation for k = 5 and k = 10.
It also derives the plot for Ridge regression that shows different polynomial degrees are affected by penalty parameter lambda 

x and y arrays were generated using make_data function from general_functions.py, 
Test MSE from bootstrap resampling was derived using bootstrap_resampling function from resampling_methods.py
Test MSE and Ridge regression as a function of both polynomial degree and penalty parameter (lambda) was derived from cross_validation function from resampling.py
"""

from general_functions import make_data, np, FIG_DIR
from resampling_methods import bootstrap_resampling, cross_validation
from plots import plt

"""Analysis of the MSE as a function of polynomial degree for Bootstrap and Cross-validation"""
mindegree, maxdegree = 1, 16
degrees = np.arange(mindegree, maxdegree)
sigma = 0.1
x,y = make_data(noise=sigma)   #default data

mse_cv_5_OLS, std_cv_5_OLS = cross_validation(x,y, model='OLS', mindegree=mindegree, maxdegree=maxdegree, k=5, return_std=True)
mse_cv_10_OLS, std_cv_10_OLS = cross_validation(x,y, model='OLS',mindegree=mindegree, maxdegree=maxdegree, k=10, return_std=True)
bootstrap_compare, *_ = bootstrap_resampling(x, y, mindegree=mindegree, maxdegree=maxdegree)

#Compare bootstrap and CV
fig, ax = plt.subplots()
ax.set_title('Mean squared error comparison different methods')
ax.plot(degrees, bootstrap_compare, 'o-', color='#FF2B2B', label='Bootstrap')
ax.axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')
#LLM Assisted from this comment

#degrees -0.1 and 0.1 so it is easier to see the standard deviation for CV k=5 and k=10 since theyre on the same degrees
ax.errorbar(degrees - 0.1, mse_cv_5_OLS, yerr=std_cv_5_OLS, fmt='s-', color='#0F2D9A',
            capsize=3, label=r'CV k=5 ($\pm$1 std)')
ax.errorbar(degrees + 0.1, mse_cv_10_OLS, yerr=std_cv_10_OLS, fmt='d-', color='#FB7100',
            capsize=3, label=r'CV k=10 ($\pm$1 std)')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE')
ax.set_xticks(degrees)
ax.set_yscale('log')
ax.legend()
plt.savefig(FIG_DIR/'Part_d_MSE_comparison.pdf')
plt.show()

results = {'Bootstrap': (bootstrap_compare, None), 'CV k=5': (mse_cv_5_OLS, std_cv_5_OLS), 'CV k=10': (mse_cv_10_OLS,std_cv_10_OLS)}
best_degrees = {}

for method, (meansq,std) in results.items():
    best_index = np.argmin(meansq)
    best_degrees[method] = degrees[best_index]
    line = f'{method:6s}: best degree = {degrees[best_index]} with MSE = {meansq[best_index]:4f} '
    if std is not None:
        line += f' ± {std[best_index]:.4f}'
    print(line)
#LLM Assited until this comment



""""
Analysis of Ridge Regression for Cross-validation method over different lambdas
Divided degrees into two for figure to be clearer and not so compact
"""
n = 100
lambdas = np.logspace(-8,4,n)

mse_polynomials_5 = np.zeros((len(lambdas), len(degrees)))

print('K=5')
for j, lmb in enumerate(lambdas):
    mse_cv_5 = cross_validation(x, y, model='Ridge', lamba=lmb, mindegree=mindegree, maxdegree=maxdegree, k=5)
    #fills current lambda row and fills it with mse values for that row
    mse_polynomials_5[j, :] = mse_cv_5 

#LLM Assisted FROM HERE
for index_degree, deg in enumerate(degrees):
    best = np.argmin(mse_polynomials_5[:, index_degree])
    print(f"  degree = {deg:2d}:  best_lambda = {lambdas[best]:8.4g}"
          f"   CV-MSE = {mse_polynomials_5[best, index_degree]:.4f}")
#LLM Assisted UNTIL HERE

mse_polynomials_10 = np.zeros((len(lambdas), len(degrees)))

print('\n','K=10')
for j, lmb in enumerate(lambdas):
    mse_cv_10 = cross_validation(x, y, model='Ridge', lamba=lmb, mindegree=mindegree, maxdegree=maxdegree, k=10)
    #fills current lambda row and fills it with mse values for that row
    mse_polynomials_10[j, :] = mse_cv_10

#Reused LLM Assisted code for k = 10
for index_degree, deg in enumerate(degrees):
    best = np.argmin(mse_polynomials_10[:, index_degree])
    print(f"  degree = {deg:2d}:  best_lambda = {lambdas[best]:8.4g}"
          f"   CV-MSE = {mse_polynomials_10[best, index_degree]:.4f}")

plotting_degrees = [4,8,12,15]
colors= ['#0FFFFF', "#FF4726", "#00AB1C", "#4000A1"]

fig, ax = plt.subplots()
for degree, color in zip(plotting_degrees, colors):
    i = np.where(degrees == degree)[0][0]          # LLM Assisted column index for this degree
    ax.plot(lambdas, mse_polynomials_5[:,i], ls='-', color=color, label=f'Degree {degree}')
    ax.plot(lambdas, mse_polynomials_10[:, i], ls='--', color=color, label=f'Degree={degree}')

ax.axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('MSE (CV)')
ax.set_title(r'Ridge CV-MSE over $\lambda$s. k = 5 (solid) and k = 10 (dashed)')
ax.legend()
plt.savefig(FIG_DIR/'Part_d_Ridge_k5_vs_k10_lambda.pdf')
plt.show()