from general_functions import make_data, np, scaling
from resampling_methods import bootstrap_resampling, cross_validation
from regression_methods import regression
import matplotlib.pyplot as plt

mindegree, maxdegree = 1, 21
x_axis_20 = np.arange(mindegree, maxdegree)
X_,y_ = make_data()
X_train, y_train, X_test, y_train = scaling(X_,y_, split_data = True )

"""
Using https://scikit-learn.org/stable/modules/cross_validation.html as a guide.
Splitting into test and train sets, where test it 0.2 of the total data

"""

# mse_cv_5_OLS = cross_validation(x,y, model='OLS',k=5)
# mse_cv_10_OLS = cross_validation(x,y, model='OLS', k=10)

# mse_cv_5_Ridge = cross_validation(x,y, model='Ridge',k=5)
# mse_cv_10_Ridge = cross_validation(x,y, model='Ridge', k=10)

# mse_cv_5_Lasso = cross_validation(x,y, model='Lasso',k=5)
# mse_cv_10_Lasso = cross_validation(x,y, model='Lasso', k=10)

lamba = 1e-3
lambas = np.logspace(-6,-1,10)
#Compare bootstrap, OLS and CV
fig, ax = plt.subplots(figsize=(8,4))
ax.set_title(r'Mean squared error comparison different methods, $\lambda$ =' + f'{lamba}')

for i in len(lambas.size):
    for model, color in zip(['OLS','Ridge','Lasso'], ['b', 'g', 'r']):
        for k, linestyle in zip([5,10], ['dashed', 'solid']):
            mse = cross_validation(X_train,y_train, model=model ,k=k, lamba = lams)
            ax.plot(x_axis_20, mse, color = color, linestyle = linestyle, label=f'{model}, k={k}')
            print(f"Finished {model} {k}")
# ax.plot(x_axis_20,mse_cv_10_OLS, 'd-', label='Cross value MSE kfold=10')

for model, color in zip(['OLS'], ['b']):
    for k, linestyle in zip([5,10], ['dashed', 'solid']):
        mse = cross_validation(X_train,y_train, model=model ,k=k, lamba = 0)
        ax.plot(x_axis_20, mse, color = color, linestyle = linestyle, label=f'{model}, k={k}')
        print(f"Finished {model} {k}")


ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE')
ax.set_xticks(x_axis_20)
ax.set_yscale('log')
ax.grid(which='both', alpha= 0.5)
ax.legend()
plt.savefig(fname="figs/Part_i_MSE_comparison.pdf")
plt.show()