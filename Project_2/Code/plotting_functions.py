from general_functions import MnistImportAndScale
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sea

from sklearn.metrics import (confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report as sk_classification_report,
                             roc_curve as sk_roc_curve, roc_auc_score, RocCurveDisplay)    

from plot_colors import *
set_style()




def plot_grid_search_results(fig, axes, search):
    """Plot cross-validation and training error for a two-parameter grid.

    The grid must contain exactly one parameter ending in ``gamma`` and one
    ending in ``lmbd``. The function displays the minimum cross-validation
    error, points within one standard error of that minimum, and the
    one-standard-error choice.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the supplied axes.
    axes : sequence of matplotlib.axes.Axes
        Two axes: the first for cross-validation results and the second for
        training results.
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search created with ``scoring="neg_mean_squared_error"``
        and ``return_train_score=True``. A single parameter-grid dictionary
        with numeric ``gamma`` and ``lmbd`` values is required.

    Returns
    -------
    dict
        Cross-validation scores, standard errors, training scores, selected
        parameters, figure, and axes.

    Examples
    --------
    Create the figure and plot a fitted grid search::

        fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
        results = plot_grid_search_results(fig, axes, search)
        argshelper.plot_or_save("grid_search_results")

    
    LLM assisted
    """

    if not hasattr(search, "cv_results_"):
        raise ValueError("Pass a fitted GridSearchCV object.")

    results = search.cv_results_
    if "mean_train_score" not in results:
        raise ValueError(
            "The search must have been created with return_train_score=True."
        )

    # Find the grid parameter names, e.g. "neuralnetregressor__gamma".
    grid = search.param_grid
    if isinstance(grid, list):
        raise ValueError("This function expects one parameter-grid dictionary.")

    gamma_keys = [key for key in grid if key.split("__")[-1] == "gamma"]
    lambda_keys = [key for key in grid if key.split("__")[-1] == "lmbd"]

    if len(gamma_keys) != 1 or len(lambda_keys) != 1:
        raise ValueError(
            "Expected exactly one gamma parameter and one lmbd parameter "
            "in search.param_grid."
        )

    gammas = list(grid[gamma_keys[0]])
    lambdas = list(grid[lambda_keys[0]])
    shape = (len(gammas), len(lambdas))

    score = -results["mean_test_score"].reshape(shape)
    train_score = -results["mean_train_score"].reshape(shape)

    n_splits = search.n_splits_
    se = results["std_test_score"].reshape(shape) / np.sqrt(n_splits - 1)

    # Find the minimum and values within one standard error of it.
    i_min, j_min = np.unravel_index(np.argmin(score), score.shape)
    within = score <= score[i_min, j_min] + se[i_min, j_min]

    # One-SE rule: choose the largest eligible lambda.
    eligible_lambda_indices = np.flatnonzero(within.any(axis=0))
    j_oser = max(eligible_lambda_indices, key=lambda j: lambdas[j])
    i_oser = np.argmin(np.where(within[:, j_oser], score[:, j_oser], np.inf))

    labels = np.array([
        [
            rf"{score[i, j]:.3f} $\pm$ {se[i, j]:.3f}"
            for j in range(len(lambdas))
        ]
        for i in range(len(gammas))
    ])

    for ax in axes:
        ax.grid(False)

    sea.heatmap(
        score,
        ax=axes[0],
        annot=labels,
        fmt="",
        cmap="plasma",
        xticklabels=[f"{value:.1e}" for value in lambdas],
        yticklabels=[f"{value:g}" for value in gammas],
        cbar_kws={"label": "SCORE"},
    )
    axes[0].set_title(f"{n_splits}-fold CV COST ± SE")
    axes[0].set_ylabel(r"$\gamma$")

    for i, j in zip(*np.where(within)):
        axes[0].add_patch(
            plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="#FF0000", lw=2)
        )

    #border around min
    axes[0].add_patch(
        plt.Rectangle((j_min, i_min), 1, 1, fill=False, edgecolor="#00AA39", lw=3)
    )

    #border around one standard error rule
    axes[0].add_patch(
        plt.Rectangle(
            (j_oser, i_oser), 1, 1, fill=False,
            edgecolor="#C9009D", lw=3, ls="--"
        )
    )

    sea.heatmap(
        train_score,
        ax=axes[1],
        annot=True,
        fmt=".3f",
        cmap="plasma",
        xticklabels=[f"{value:.1e}" for value in lambdas],
        yticklabels=[f"{value:g}" for value in gammas],
        cbar_kws={"label": "SCORE"},
    )
    axes[1].set_title("Training SCORE")

    for ax in axes:
        ax.set_xlabel(r"$\lambda$")

    fig.tight_layout()

    return {
        "score": score,
        "se": se,
        "train_score": train_score,
        "within_one_se": within,
        "best_params": search.best_params_,
        "one_se_params": {
            gamma_keys[0]: gammas[i_oser],
            lambda_keys[0]: lambdas[j_oser],
        },
        "figure": fig,
        "axes": axes,
    }



def plot_training_history(fig, ax, search, metric="MSE", label = 'Train', color = COLORS['nn']):
    """Plot the training history of a fitted search's refit estimator.

    If the best estimator is a pipeline, its final step is used. The network
    architecture is inferred from the fitted weight matrices and shown in
    the title. Validation history is plotted when available.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes
        Axes on which to draw the history.
    search : sklearn.model_selection.GridSearchCV
        Fitted search with ``refit=True``. Its best estimator must expose
        ``history`` and ``NN.layers``.
    metric : str, default="MSE"
        Label for the vertical axis.
    label : str, default="Train"
        Legend label for the training curve.
    color : color, default=COLORS["nn"]
        Color used for the training and validation curves.

    Returns
    -------
    tuple
        The figure and axes.

    Examples
    --------
    Plot the training history of a fitted search::

        fig, ax = plt.subplots()
        plot_training_history(fig, ax, search, metric="MSE")
        argshelper.plot_or_save("training_history")


    LLM assisted
    """

    if not hasattr(search, "best_estimator_"):
        raise ValueError("The search must be fitted with refit=True.")

    model = search.best_estimator_

    # If the best estimator is a Pipeline, get its final estimator.
    if hasattr(model, "named_steps"):
        model = list(model.named_steps.values())[-1]

    if not hasattr(model, "history") or not hasattr(model, "NN"):
        raise ValueError("The best estimator does not expose training history and network layers.")

    history = model.history
    weights = model.NN.layers

    layer_sizes = [weights[0][0].shape[0]]
    layer_sizes.extend(W.shape[1] for W, _ in weights)
    architecture = "[" + ", ".join(map(str, layer_sizes)) + "]"

    epochs = history.get("epoch", np.arange(1, len(history["train"]) + 1))


    ax.plot(epochs, history["train"], TRAIN_LS, color=color, label=label)

    if "val" in history:
        if len(history["val"]) != 0:
            ax.plot(epochs, history["val"], VAL_LS, color=color,
                alpha=0.5, label="Validation")
        else:
            pass

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_yscale("log")
    ax.set_title(f"Training history, network {architecture}")
    ax.legend()
    fig.tight_layout()
    return fig, ax



def plot_three_parameter_grid(fig, ax, search, parameter_keys=None):
    """Plot a fitted three-parameter grid search in 3D.

    The selected numeric parameters are shown on the x, y, and z axes. Mean
    cross-validation scores are encoded by point color using ``MSE_CMAP``;
    the best result is marked with a star. Scores are displayed as returned
    by scikit-learn, so higher values indicate better results (including for
    scorers such as ``"neg_mean_squared_error"``).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the supplied 3D axes.
    ax : matplotlib.axes.Axes
        Three-dimensional axes, typically created with
        ``fig.add_subplot(111, projection="3d")``.
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search with a single parameter-grid dictionary.
    parameter_keys : sequence of str or None, default=None
        Exactly three numeric parameter keys from ``search.param_grid``.
        If omitted, all parameter keys are used, so the grid must contain
        exactly three parameters. Pipeline parameter names should be passed
        in full, such as ``"neuralnetregressor__gamma"``.

    Returns
    -------
    dict
        Mean cross-validation scores, best parameters, figure, and axes.

    Examples
    --------
    Plot a fitted search with exactly three numeric grid parameters::

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        results = plot_three_parameter_grid(fig, ax, search)
        plt.show()

    Select three parameters from a grid with additional parameters::

        keys = [
            "neuralnetregressor__gamma",
            "neuralnetregressor__lmbd",
            "neuralnetregressor__batch_size",
        ]
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        results = plot_three_parameter_grid(
            fig, ax, search, parameter_keys=keys
        )
        plt.show()
    
    LLM assisted
    """
    if not hasattr(search, "cv_results_"):
        raise ValueError("Pass a fitted GridSearchCV object.")


    if isinstance(search.param_grid, list):
        raise ValueError("This function expects one parameter-grid dictionary.")

    if parameter_keys is None:
        parameter_keys = list(search.param_grid)

    if len(parameter_keys) != 3:
        raise ValueError("Provide exactly three parameter keys to plot.")

    rows = search.cv_results_["params"]
    try:
        values = [
            np.asarray([row[key] for row in rows], dtype=float)
            for key in parameter_keys
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The selected grid parameters must all be numeric.") from exc

    cv_score = np.asarray(search.cv_results_["mean_test_score"])

    points = ax.scatter(
        *values,
        c=cv_score,
        cmap=MSE_CMAP,
        edgecolors=COLORS["text"],
    )

    best_index = search.best_index_
    ax.scatter(
        *(value[best_index] for value in values),
        color=COLORS["nn"],
        marker="*",
        s=180,
        edgecolors=COLORS["text"],
        label="Best CV score",
    )

    ax.set_xlabel(parameter_keys[0].split("__")[-1])
    ax.set_ylabel(parameter_keys[1].split("__")[-1])
    ax.set_zlabel(parameter_keys[2].split("__")[-1])
    ax.set_title("Grid-search CV score")
    ax.legend()

    fig.colorbar(points, ax=ax, label="Mean CV score")
    fig.tight_layout()

    return {
        "cv_score": cv_score,
        "best_params": search.best_params_,
        "figure": fig,
        "axes": ax,
    }