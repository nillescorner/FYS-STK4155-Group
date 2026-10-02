from general_functions import make_data, np, scaling, FIG_DIR
from resampling_methods import bootstrap_resampling, cross_validation
from regression_methods import regression
import matplotlib.pyplot as plt

mindegree, maxdegree = 1, 21
x_axis_20 = np.arange(mindegree, maxdegree)
X_,y_ = make_data()
X_train, X_test, y_train, y_test = scaling(X_,y_, split_data = True )

"""
Using https://scikit-learn.org/stable/modules/cross_validation.html as a guide.
Splitting into test and train sets, where test it 0.2 of the total data

lambdas when comparing the estimators is not directly equivalent works differently for each model

Basic outline of this part:

1. Do CV for model and penalty and deg for full visualisation of interplay
2. Do GridSearch to find best suited hyperparams (in our case the penalty lambda)
3. Retrain models per degree per best suited hypermaram
4. Do MSE-decomposition to get a full picture on the bias-variance tradeoff


"""

# mse_cv_5_OLS = cross_validation(x,y, model='OLS',k=5)
# mse_cv_10_OLS = cross_validation(x,y, model='OLS', k=10)

# mse_cv_5_Ridge = cross_validation(x,y, model='Ridge',k=5)
# mse_cv_10_Ridge = cross_validation(x,y, model='Ridge', k=10)

# mse_cv_5_Lasso = cross_validation(x,y, model='Lasso',k=5)
# mse_cv_10_Lasso = cross_validation(x,y, model='Lasso', k=10)

lamba = 1e-3
lambas = np.logspace(-3,3,9) #penalties to apply to estimators
#Compare bootstrap, OLS and CV
fig, axs = plt.subplots(3,3, figsize=(8*3,4*3),sharey = True, sharex = True)
axs = axs.flat #flattening axes
fig.suptitle(r'Mean squared error comparison different methods, $\lambda$ =' + f'{lamba}')

for i in range(lambas.size):
    for k, linestyle in zip([5,10], ['dashed', 'solid']):
        for model, color in zip(['Ridge','Lasso'], ['g', 'r']):
            #calculating and plotting ridge and lasso
            mse = cross_validation(X_train,y_train, model=model ,k=k, lamba = lambas[i])
            axs[i].plot(x_axis_20, mse, color = color, linestyle = linestyle, label=f'{model}, k={k}')
            axs[i].set_xticks(x_axis_20)
            axs[i].set_title(r"$\lambda$ = " + f"{lambas[i]}")
            print(f"Finished {model=}, {k=}, lambda = {lambas[i]}")

        #calculating and plotting ols 
        mse_OLS = cross_validation(X_train,y_train, model='OLS' ,k=k, lamba = 0)
        axs[i].plot(x_axis_20, mse_OLS, color = color, linestyle = linestyle, label=f'{model}, k={k}')
        print(f"Finished model='OLS', {k=}, lambda = {lambas[i]}")
        axs[i].grid(which='both', alpha= 0.5)



axs[0].set_yscale('log')
fig.supxlabel('Polynomial Degree')
fig.supylabel('MSE')
axs[0].legend()
plt.savefig(FIG_DIR/"Part_i_MSE_comparison_lams.pdf")
plt.show()



# ax.set_xlabel('Polynomial Degree')
# ax.set_ylabel('MSE')
# ax.set_xticks(x_axis_20)
# ax.set_yscale('log')
# ax.grid(which='both', alpha= 0.5)
# ax.legend()
# plt.savefig(FIG_DIR/"Part_i_MSE_comparison.pdf")
# plt.show()