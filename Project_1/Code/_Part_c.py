from regression_methods import regression
from general_functions import make_data,np
from plots import plot_regression, plt
from resampling_methods import bootstrap_resampling


x,y = make_data()   #default data

mindegree, maxdegree = 1, 21
mse_train_fig, mse_test_fig, *_ = regression(x,y, maxdeg=maxdegree)
x_axis_20 = np.arange(mindegree,maxdegree)

plot_regression(x_axis_20, mse_train_fig, mse_test_fig, titles='Recreating Fig 2.11 with OLS using MSE train and test data', yscale='log')

error, bias, variance = bootstrap_resampling(x,y,mindegree=mindegree, maxdegree=maxdegree)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x_axis_20, error, 'o-', label='Test Error')
ax.plot(x_axis_20, bias, 's-', label=r'Bias$^2$ (+ $\sigma^2$)')
ax.plot(x_axis_20, variance, 'd-', label='Variance')
ax.set_yscale('log')
ax.set_title('Bias-variance decomposition via bootstrap')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Error')
ax.legend()
ax.set(xticks=x_axis_20)
plt.show()


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x_axis_20, error, 'o-', label='Test MSE Bootstrap')
ax.plot(x_axis_20, bias, 's-', label=r'Bias$^2$ (+ $\sigma^2$) Bootstrap')
ax.plot(x_axis_20, variance, 'd-', label='Variance Bootstrap')
ax.plot(x_axis_20, mse_test_fig, 'o-', label='Testing MSE OLS' )
ax.set_yscale('log')
ax.set_title('Bias-variance decomposition')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Error')
ax.legend()
ax.set(xticks=x_axis_20)
plt.show()


ns = [40,100,400]
fig, ax = plt.subplots(1,3, figsize=(16, 4), sharey=True)
ax.flatten()
for i,n in enumerate(ns):
    x_n, y_n = make_data(n=n)
    error, bias, variance = bootstrap_resampling(x_n,y_n)
    ax[i].plot(x_axis_20, error, 'o-', color='r', label='Test Error')
    ax[i].plot(x_axis_20, bias, 's-', color='b', label=r'Bias$^2$ (+ $\sigma^2$)')
    ax[i].plot(x_axis_20, variance, 'd-', color='g', label='Variance')

    ax[i].set_yscale('log')
    ax[i].set_xlabel('Polynomial Degree')
    ax[i].set(xticks=x_axis_20)
    ax[i].set_title(f'Bias-variance decomposition for n = {n}')

ax[0].set_ylabel('MSE Decomposition')
ax[0].legend()
plt.tight_layout()
plt.show()