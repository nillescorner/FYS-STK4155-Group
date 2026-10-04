from general_functions import make_data, np, scaling, FIG_DIR, train_test_split, runge
from resampling_methods import bootstrap_resampling, cross_validation, bootstrap_resampling, train_through_gridsearchCV, mse_decomposer
from regression_methods import regression
from sklearn.preprocessing import PolynomialFeatures

# for plotting
import matplotlib.pyplot as plt
import colors as c
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mindegree, maxdegree = 1, 21
x_axis_20 = np.arange(mindegree, maxdegree)
X_,y_ = make_data()
X_train, X_test, y_train, y_test = scaling(X_,y_, split_data = True )

"""
This is implements the functions defined in resampling_methods 

Basic outline of this part:

1. Do CV for model and penalty and deg for full visualisation of interplay
2. Do GridSearch to find best suited hyperparams (in our case the penalty lambda)
3. Retrain models per degree per best suited hypermaram
4. Do MSE-decomposition to get a full picture on the bias-variance tradeoff

NOTE: MSE decomposition has to be toggled below. It is slow


"""

### Toggles for parts of the code. It is slow as shit

bootstrap_kfold_resampling_MSE_decomposition = True #only run it if you have to it takes ~7 min

###


mindegree, maxdegree = 1, 21

degs = np.arange(mindegree, maxdegree)
x_,y_ = make_data()
x_train, x_test, y_train, y_test = train_test_split(x_,y_, test_size=0.2, random_state=2026)


results, model_shorthands = train_through_gridsearchCV(
    x_, y_, models=['ridge', 'lasso'], Ks=[5, 10], return_std=True
)



"""

Plot of tuned hyperparameter per degree
"""

fig, ax = plt.subplots(figsize=(5, 5))


for model_shorthand in model_shorthands:
    model, _, _, k = model_shorthand.split()

    if model == 'ridge':
        color = c.RIDGE if k == '5' else c.RIDGE2
    elif model == 'lasso':
        color = c.LASSO if k == '5' else c.LASSO2

    filled_marker_style = dict(
        marker='*',
        markersize=15,
        linestyle='none',
        color='darkgrey',
        markerfacecolor="#FAFE81",
        markeredgecolor=color,
    )

    ax.plot(
        degs,
        results[model_shorthand]['best_params'],
        color=color,
        label=model.capitalize() + f" k = {k}",
    )

    best_idx = np.argmin(results[model_shorthand]['mse'])
    ax.plot(
        degs[best_idx],
        results[model_shorthand]['best_params'][best_idx],
        label=(
            r'Best $\lambda$ ='
            + f"{results[model_shorthand]['best_params'][best_idx]:0.2e}"
        ),
        **filled_marker_style,
    )

ax.set_title(r'Best parameters $\lambda$ per degree')
ax.set_ylabel(r'Penalty $\lambda$')
ax.set_xticks(degs)
ax.grid(which="both", alpha=0.5)
ax.set_yscale("log")
ax.legend()

fig.supxlabel("Polynomial Degree").set_in_layout(True)
fig.tight_layout()
plt.savefig(fname=FIG_DIR/'Part_i_besthyperparameters.pdf')



"""

Plotting for MSE with std vs test MSE

"""

fig, axs = plt.subplots(5,1, figsize = (5, 14))   
axs = axs.flat
# Using OLS as a reference 
import colors as c
idx = 4
for k, linesstyle, color in zip([5,10], ['dotted','dashed'], [c.OLS, c.OLS2]):
        mse,  std= cross_validation(
                x_train, y_train, model="OLS", k=k, return_std=True)
        mse, std = np.array(mse), np.array(std)
        axs[idx].plot(degs, mse, color = color,  label = f'OLS k = {k}')
        axs[idx].fill_between(degs,
                                        mse - std,
                                        mse + std,
                                        alpha=0.15,
                                        color=color,
                                        )
        axs[idx].set_xticks(degs)
        axs[idx].grid(which="both", alpha=0.5)
        axs[idx].set_yscale("log")
        axs[idx].set_title(r'OLS as reference ($\lambda$ not tuned)')

        best_idx = np.argmin(mse)
        filled_marker_style = dict(marker='*', markersize=15, linestyle = 'none',
                                   color='darkgrey',
                                   markerfacecolor="#FAFE81",
                                   markeredgecolor=color)
        
        axs[idx].plot(degs[best_idx], mse[best_idx],
                        label = f'Best score = {mse[best_idx]:0.2e}',  **filled_marker_style) 
        
        axs[idx].legend(framealpha=0.3)
    


for idx, model_shorthand in zip(range(0,4,1), model_shorthands):
        model ,_ ,_ ,k = model_shorthand.split()

        if model == 'ridge':
                if k == '5':
                        color = c.RIDGE
                else:
                        color = c.RIDGE2
        elif model == 'lasso':
                if k == '5':
                        color = c.LASSO
                else:
                        color = c.LASSO2


    
    
        axs[idx].plot(degs, results[model_shorthand]['test_mse'], color = color, linestyle = 'dashed',  label =  model + f" k={k} test")
        # axs[idx].errorbar(degs, results[model_shorthand]['mse'], yerr= std_uncertainty(results[model_shorthand]['std_test_score'],k),
        #                 color = color, linestyle = 'solid',  label = model_shorthand + " k-fold MSE", capsize = 5)
        axs[idx].plot(degs, results[model_shorthand]['mse'],
                        color = color, linestyle = 'solid',  label =  model + f" k={k}")

        axs[idx].fill_between(degs,
                                results[model_shorthand]['mse'] - results[model_shorthand]['std_test_score'],
                                results[model_shorthand]['mse'] + results[model_shorthand]['std_test_score'],
                                alpha=0.15,
                                color=color,
                                )
        
        

        filled_marker_style = dict(marker='*', markersize=15, linestyle = 'none',
                           color='darkgrey',
                           markerfacecolor="#FAFE81",
                           markeredgecolor=color)

        best_idx = np.argmin(results[model_shorthand]['mse'])

        axs[idx].plot(degs[best_idx], results[model_shorthand]['mse'][best_idx],
                         label = f'Best score = {results[model_shorthand]['mse'][best_idx]:0.2e}',  **filled_marker_style) 

        axs[idx].set_ylim(4e-3,3e-1)
        axs[idx].set_yscale("log")

        axs[idx].set_xticks(degs)
        axs[idx].grid(which="both", alpha=0.5)
        axs[idx].legend(framealpha=0.5)

    
    


fig.suptitle(r'Best CV-score and test score for tuned $\lambda$')
fig.supxlabel("Polynomial Degree").set_in_layout(True)
fig.supylabel("MSE").set_in_layout(True)
# fig.supylabel("Cross-validated MSE").set_in_layout(True)
fig.tight_layout()
plt.savefig(fname=FIG_DIR/'Part_i_MSE_testandtrain.pdf')


"""
Model fit to data and true runge function

"""


def plot_best_model_fits(x, y, results, model_shorthands, degrees, ncols, nrows, figsize):
    """Plot and return the fitted GridSearchCV model selected for each configuration.
        LLM used for plotting function, but logic defined by ourselves.

        Params:
                x (array): Input values used to generate the data.
                y (array): Target values corresponding to x.
                results (dict): Results returned by train_through_gridsearchCV.
                        Each model and fold-count entry must contain fitted GridSearchCV
                        objects indexed by polynomial degree and their cross-validation MSEs.
                model_shorthands (list): Model and fold-count keys used to index
                        results, such as "ridge folds = 5".
                degrees (array): Polynomial degrees corresponding to the MSE values.
                ncols (int): Number of subplot columns. If ncols and nrows are both
                        1, all model fits are plotted on the same axes.
                 
                nrows (int): Number of subplot rows. If ncols and nrows are both
                        1, the generated data and Runge function are plotted only once.
                figsize (tuple): Figsize as used in subplot 

        Returns:
                best_models (dict): Fitted GridSearchCV objects for the degree with
                        the lowest cross-validation MSE for each model and fold-count
                        configuration, indexed by model shorthand.

        Saves:
                A plot of the selected model fits to
                FIG_DIR / "Part_i_bestfits.pdf".

    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).ravel()
    x_sorted = x[np.argsort(x[:, 0])]

    colors = {
        ('ridge', '5'): c.RIDGE,
        ('ridge', '10'): c.RIDGE2,
        ('lasso', '5'): c.LASSO,
        ('lasso', '10'): c.LASSO2,
    }

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    best_models = {}
    single_axes = ncols == 1 and nrows == 1

    if single_axes:
        ax = axs
        axes_and_models = ((ax, model_shorthand) for model_shorthand in model_shorthands)

        # Shared data and reference function are plotted once on the combined axes.
        shared_handles = [
            ax.scatter(x[:, 0], y, color="#D8A35A", s=16, alpha=0.65, label='Generated data'),
            ax.plot(
                x_sorted[:, 0],
                runge(x_sorted[:, 0]),
                color='black',
                linestyle='dashed',
                label='Runge function',
            )[0],
        ]
        fit_handles = []
    else:
        axes = np.asarray(axs, dtype=object).reshape(-1)
        axes_and_models = zip(axes, model_shorthands)

    for ax, model_shorthand in axes_and_models:
        model_name, _, _, k = model_shorthand.split()
        color = colors[(model_name, k)]

        best_idx = np.argmin(results[model_shorthand]['mse'])
        best_degree = int(degrees[best_idx])
        best_model = results[model_shorthand]['fit'][best_degree]
        best_models[model_shorthand] = best_model

        X_sorted = PolynomialFeatures(degree=best_degree).fit_transform(x_sorted)
        y_pred = best_model.predict(X_sorted).ravel()
        alpha = best_model.best_params_[f'{model_name}__alpha']

        if not single_axes:
            ax.scatter(
                x[:, 0], y,
                color="#D8A35A",
                s=16,
                alpha=0.65,
                label='Generated data',
            )
            ax.plot(
                x_sorted[:, 0],
                runge(x_sorted[:, 0]),
                color='black',
                linestyle='dashed',
                label='Runge function',
            )

        fit_handle, = ax.plot(
            x_sorted[:, 0],
            y_pred,
            color=color,
            linewidth=2,
            label=(
                f'{model_name.capitalize()} k={k}, '
                + r'$\lambda$'
                + f'={alpha:.2e}'
            ),
        )
        if single_axes:
            fit_handles.append(fit_handle)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(alpha=0.4)
        if not single_axes:
            ax.legend()

    if single_axes:
        fit_legend = ax.legend(handles=fit_handles, loc='lower center', title='Model fits', framealpha = 0.3)
        ax.add_artist(fit_legend)
        ax.legend(handles=shared_handles, loc='upper right', title='Reference',  framealpha = 0.3)

    fig.suptitle('Best Ridge and Lasso fits by cross-validation MSE')
    fig.tight_layout()
    plt.savefig(fname=FIG_DIR / 'Part_i_bestfits.pdf')
    return best_models


best_models = plot_best_model_fits(
    x_, y_, results, model_shorthands, degs, 1, 1, (5,4)
)


"""
Bootstrap and kfold resampling for MSE decomposition
Takes ~7 min
"""

if bootstrap_kfold_resampling_MSE_decomposition:

        kfold_resamples = 40
        bootstrap_resamples = 50
        mse_decomposition_kfold = mse_decomposer(x_,y_, results, model_shorthands, resamples = kfold_resamples)
        mse_decomposition_bootstrap  = mse_decomposer(x_,y_, results, model_shorthands, resamples = bootstrap_resamples, method = "bootstrap_resampling")


        fig, axs = plt.subplots(1*len(model_shorthands),1, figsize = (5,15), sharey = True)   
        # fig, axs = plt.subplots(1,1*len(model_shorthands), figsize = (5*len(model_shorthands), 5), sharey = True)   
        # fig, axs = plt.subplots(2,2, figsize = (10, 10))   
        # axs = axs.flat
        colors = c.ERROR, c.BIAS, c.VARIANCE
        error_bias_variance = ['error','bias','variance']
        for idx, model_shorthand in enumerate(model_shorthands):
                ax = np.atleast_1d(axs)[idx]  # handles the single-Axes case from plt.subplots

                
        
                for i, mse_decomposition in enumerate([mse_decomposition_bootstrap, mse_decomposition_kfold]):
                        model ,_ ,_ ,k = model_shorthand.split()
                

                        if i == 0:
                                resamplinglinestyle = "dotted"
                                label = 'bootstrap'
                        elif i == 1:
                                resamplinglinestyle = 'solid'
                                label = 'kfold'

                        if model == 'ridge':
                                if k == '5':
                                        color = c.RIDGE
                                else:
                                        color = c.RIDGE2
                        elif model == 'lasso':
                                if k == '5':
                                        color = c.LASSO
                                else:
                                        color = c.LASSO2

                        

                        
                        ax.plot(degs, mse_decomposition[model_shorthand]['error'],
                                color = c.ERROR, linestyle = resamplinglinestyle)
                        ax.plot(degs, mse_decomposition[model_shorthand]['variance'],
                                color = c.BIAS, linestyle = resamplinglinestyle)
                        ax.plot(degs, mse_decomposition[model_shorthand]['bias2_plus_noise'], 
                                color = c.VARIANCE, linestyle = resamplinglinestyle)
                        
                        ax.set_yscale("log")
                        ax.set_title(model_shorthand)
                        ax.set_xticks(degs)
                        ax.grid(which="both", alpha=0.5)
                        ax.set_title(f"{model.capitalize()} k = {int(k)}" )

                ax.plot(degs, results[model_shorthand]['test_mse'], color = 'black', linestyle = 'dashed',  label = "test MSE")
                ax.plot(degs, results[model_shorthand]['mse'], color = 'black', linestyle = 'solid',  label = "MSE")

                

        bias_variance_handels = [
        Line2D([0], [0],  linestyle="solid", color = c.ERROR, linewidth=2, label="error"),
        Line2D([0], [0],  linestyle="solid", color = c.BIAS, linewidth=2, label="variance"),
        Line2D([0], [0],  linestyle="solid", color = c.VARIANCE, linewidth=2, label=r"$bias^2 + \sigma^2_\eta$"),
        ]


        # --- keep MSE/test MSE labels too ---
        main_handles = [
        Line2D([0], [0], color="black", linestyle="solid", linewidth=2, label="MSE"),
        Line2D([0], [0], color="black", linestyle="dashed", linewidth=2, label="test MSE"),
        ]

        kfold_bootstrap_handels = [
                Line2D([0], [0], color="grey", linestyle="solid", linewidth=2, label="Kfold"),
                Line2D([0], [0], color="grey", linestyle="dotted", linewidth=2, label="bootstrap"),
        ]

        legend1 = axs[0].legend(
        handles=kfold_bootstrap_handels ,
        loc="upper left",
        frameon=True,
        title = "linestyles represent", 
        ncol=2,
        fontsize=10,
        )
        axs[0].add_artist(legend1)

        legend2 = axs[0].legend(
        handles=main_handles ,
        loc="lower right",
        frameon=True,
        title = "MSE linestyle and color by", 
        ncol=2,
        fontsize=10,
        )
        axs[0].add_artist(legend2)


        legend3 = axs[0].legend(
        handles=bias_variance_handels,
        loc="lower left",
        frameon=True,
        title = "colors represent",
        fontsize=9,
        )

        fig.suptitle(f"Comparison of kfold and bootstrap resampling\n for bias variance sweep. \n {bootstrap_resamples=}, {kfold_resamples=} \n")
        fig.supxlabel("Polynomial Degree").set_in_layout(True)
        fig.supylabel("MSE").set_in_layout(True)
        fig.tight_layout()
        plt.savefig(fname=FIG_DIR/'Part_i_hyperparameters_msedecomposition_kfold_bootstrap.pdf')