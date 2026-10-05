"""
This python file contains the code used to derive the figure to replicate Fig 2.11 Hastie, Tibshirani and Friedman, 
and the figure showing bias-variance tradeoff of the Runge Function using simpler ordinary least squares and compares it 
 to test MSE from OLS regression from part a.

x and y arrays were generated using make_data function from general_functions.py, 
MSE was derived using regression function from regression_methods.py,
Bias-variance tradeoff was performed using bootstrap_resampling function from resampling_methods.py
"""

from regression_methods import regression
from general_functions import make_data,np, FIG_DIR
from plots import plt
from resampling_methods import bootstrap_resampling
sigma=0.1

x,y = make_data(n=100, noise=sigma)   #default data

mindegree, maxdegree = 1, 21    #purposely high degree to show figure better
mse_train_fig, mse_test_fig, *_ = regression(x,y, mindeg=mindegree, maxdeg=maxdegree)
degrees = np.arange(mindegree,maxdegree)

fig,ax = plt.subplots()
ax.plot(degrees,mse_train_fig, color='#8C564B', label='Training data')
ax.plot(degrees,mse_test_fig, color='#0AC23E', label='Test data')
ax.axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')

ax.set_title('Recreating Figure 2.11 with MSE from OLS Regression')
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('MSE')
ax.set_yscale('log')
ax.set_xticks(degrees)
ax.legend()
#plt.savefig(FIG_DIR/'Part_c_recreate_2.11.pdf')
plt.show()


mindegree, maxdegree = 1, 16 #same as for other parts
degrees = np.arange(mindegree, maxdegree)
error, bias, variance = np.asarray(bootstrap_resampling(x,y,mindegree=mindegree, maxdegree=maxdegree))
_, mse_test, *_ = regression(x,y, mindeg=mindegree, maxdeg=maxdegree)

best_index = np.argmin(error)
best_degree = degrees[best_index]
best_mse = error[best_index]
print(f'Best mse for bootstrap {best_mse:.3e} with bias^2: {bias[best_index]:.3e} and variance: {variance[best_index]:.3e}')
print(f'Best mse for OLS {mse_test[np.argmin(mse_test)]:.3e}')


fig, ax = plt.subplots()
ax.plot(degrees, error, 'o-', color='#FF2B2B', label='Test MSE Bootstrap')
ax.plot(degrees, bias, 's-', color='#4E20A1', label=r'Bias$^2$ (+ $\sigma^2$) Bootstrap')
ax.plot(degrees, variance, 'd-', color='#FFB107', label='Variance Bootstrap')
ax.plot(degrees, mse_test, 'o-', color='#F433DA', label='Test MSE OLS' )
ax.axvline(x=best_degree, color="#696968", ls='--', label=rf'Best degree Bootstrap = {best_degree}')
ax.axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')


ax.set_yscale('log')
ax.set_title('Bias-variance tradeoff and test MSE for OLS')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE decomposition')
ax.legend()
ax.set(xticks=degrees)
#plt.savefig(FIG_DIR/'Part_c_bootstrap.pdf')
plt.show()


"""Analysis of bootstrap for different n values"""
ns = [40,100,400]
fig, ax = plt.subplots(3,1, figsize=(7, 7), sharex=True)
ax.flatten()
for i,n in enumerate(ns):
    x_n, y_n = make_data(n=n)
    error, bias, variance = bootstrap_resampling(x_n,y_n, mindegree=mindegree, maxdegree=maxdegree)
    ax[i].plot(degrees, error, 'o-', color='#FF2B2B', label='Test Error')
    ax[i].plot(degrees, bias, 's-',  color='#4E20A1', label=r'Bias$^2$ (+ $\sigma^2$)')
    ax[i].plot(degrees, variance, 'd-', color='#FFB107', label='Variance')
    ax[i].axhline(sigma**2, color='black', ls=':', label=rf'Noise floor $\sigma^2$ = {sigma**2:.3g}')


    ax[i].set_yscale('log')
    ax[i].set_ylabel('MSE decomposition')
    ax[i].set_title(f'Bias-variance tradeoff for n = {n}')

ax[0].legend()
ax[2].set(xticks=degrees)
ax[2].set_xlabel('Polynomial Degree')

plt.tight_layout()
#plt.savefig(FIG_DIR/'Part_c_bootstrap_diff_ns.pdf')
plt.show()

