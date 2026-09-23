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
plt.show()