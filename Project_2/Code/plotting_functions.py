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
    """Plot cross-validation and training scores for a two-parameter grid.

    The grid must contain exactly one parameter ending in ``gamma`` and one
    ending in ``lmbd``. Negative MSE scores are displayed as positive MSE;
    other scores are shown as returned by scikit-learn. The one-standard-
    error region and selected point account for whether lower or higher
    scores are better.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the supplied axes.
    axes : sequence of matplotlib.axes.Axes
        Two axes: the first for cross-validation results and the second for
        training results.
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search created with ``return_train_score=True`` and a
        single parameter-grid dictionary with numeric ``gamma`` and ``lmbd``
        values. Supports ``"neg_mean_squared_error"`` and higher-is-better
        scorers such as ``"accuracy"``.

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

    is_neg_mse = search.scoring == "neg_mean_squared_error"
    score = results["mean_test_score"].reshape(shape)
    train_score = results["mean_train_score"].reshape(shape)
    if is_neg_mse:
        score = -score
        train_score = -train_score

    n_splits = search.n_splits_
    se = results["std_test_score"].reshape(shape) / np.sqrt(n_splits - 1)

    # Find the best score and the values within one standard error of it.
    if is_neg_mse:
        best_i, best_j = np.unravel_index(np.argmin(score), score.shape)
        within = score <= score[best_i, best_j] + se[best_i, best_j]
    else:
        best_i, best_j = np.unravel_index(np.argmax(score), score.shape)
        within = score >= score[best_i, best_j] - se[best_i, best_j]

    # One-SE rule: choose the largest eligible lambda.
    eligible_lambda_indices = np.flatnonzero(within.any(axis=0))
    j_oser = max(eligible_lambda_indices, key=lambda j: lambdas[j])
    if is_neg_mse:
        i_oser = np.argmin(np.where(within[:, j_oser], score[:, j_oser], np.inf))
    else:
        i_oser = np.argmax(np.where(within[:, j_oser], score[:, j_oser], -np.inf))

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
        cbar_kws={"label": "MSE" if is_neg_mse else "Score"},
    )
    axes[0].set_title(f"{n_splits}-fold CV {'MSE' if is_neg_mse else 'score'} ± SE")
    axes[0].set_ylabel(r"$\gamma$")

    for i, j in zip(*np.where(within)):
        axes[0].add_patch(
            plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="#FF0000", lw=2)
        )

    #border around min
    axes[0].add_patch(
        plt.Rectangle((best_j, best_i), 1, 1, fill=False, edgecolor="#00AA39", lw=3)
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
        cbar_kws={"label": "MSE" if is_neg_mse else "Score"},
    )
    axes[1].set_title(f"Training {'MSE' if is_neg_mse else 'score'}")

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



from types import SimpleNamespace
import numpy as np


def grid_search_slice(search, fixed_key, fixed_value):
    """Return one fixed-value slice of a fitted grid search.

    The returned view can be passed to ``plot_grid_search_results``.

    Example
    -------
    for value in search.param_grid["neuralnetregressor__batch_size"]:
        sliced = grid_search_slice(
            search, "neuralnetregressor__batch_size", value
        )
        fig, axes = plt.subplots(2, 1)
        plot_grid_search_results(fig, axes, sliced)
        plt.show()

    LLM assisted
    """
    if isinstance(search.param_grid, list):
        raise ValueError("This helper expects one parameter-grid dictionary.")
    if fixed_key not in search.param_grid:
        raise KeyError(f"{fixed_key!r} is not in search.param_grid.")

    results = search.cv_results_
    mask = np.array(
        [params[fixed_key] == fixed_value for params in results["params"]]
    )
    if not mask.any():
        raise ValueError(f"No grid results found for {fixed_key}={fixed_value!r}.")

    sliced_results = {
        "params": [params for params, keep in zip(results["params"], mask) if keep],
        "mean_test_score": results["mean_test_score"][mask],
        "std_test_score": results["std_test_score"][mask],
        "mean_train_score": results["mean_train_score"][mask],
    }

    best_index = np.argmax(sliced_results["mean_test_score"])
    return SimpleNamespace(
        cv_results_=sliced_results,
        scoring=search.scoring,
        param_grid={
            key: values
            for key, values in search.param_grid.items()
            if key != fixed_key
        },
        n_splits_=search.n_splits_,
        best_params_=sliced_results["params"][best_index],
    )


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

    When ``search.param_grid`` is a list, each grid dictionary is shown in
    its own subplot. Unless ``parameter_keys`` is provided, the three
    parameters with multiple candidates in each dictionary are plotted.
    Categorical values are mapped to labeled tick positions, and point
    colors show mean CV scores.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure for the plot.
    ax : matplotlib.axes.Axes
        A 3D axis for one grid, one 3D axis per grid dictionary, or ``None``
        to create the required subplot(s) in ``fig``.
    search : sklearn.model_selection.GridSearchCV
        Fitted grid search with one parameter-grid dictionary or a list of
        dictionaries.
    parameter_keys : sequence of str or None, default=None
        Exactly three parameter keys to plot. If omitted, the three varying
        parameters are selected automatically for each dictionary. Pipeline
        parameter names should be passed in full.

    Returns
    -------
    dict
        Mean cross-validation scores, best parameters, figure, and axes.

    Examples
    --------
    Plot a list of grids, with one subplot per dictionary::

        fig = plt.figure(figsize=(6 * len(search.param_grid), 6))
        results = plot_three_parameter_grid(fig, None, search)
        plt.show()
    
    LLM assisted
    """
    if not hasattr(search, "cv_results_"):
        raise ValueError("Pass a fitted GridSearchCV object.")

    grids = search.param_grid if isinstance(search.param_grid, list) else [search.param_grid]
    if not grids:
        raise ValueError("search.param_grid must contain at least one grid.")

    if ax is None:
        axes = np.asarray(
            fig.subplots(
                1, len(grids), subplot_kw={"projection": "3d"}, squeeze=False
            )
        ).ravel()
    elif isinstance(search.param_grid, list):
        axes = np.asarray(ax, dtype=object).ravel()
        if len(axes) != len(grids):
            raise ValueError(
                "Provide one 3D axis per parameter grid, or pass ax=None."
            )
    else:
        axes = np.asarray([ax], dtype=object)

    rows = search.cv_results_["params"]
    all_scores = np.asarray(search.cv_results_["mean_test_score"])
    score_min, score_max = np.min(all_scores), np.max(all_scores)
    plotted_scores = []
    subplot_best_params = []

    for grid_index, (grid, axis) in enumerate(zip(grids, axes)):
        keys = (
            [key for key, candidates in grid.items() if len(candidates) > 1]
            if parameter_keys is None
            else list(parameter_keys)
        )
        if len(keys) != 3:
            raise ValueError(
                f"Grid {grid_index} has {len(keys)} varying/selected parameters; "
                "exactly three are required."
            )
        missing = [key for key in keys if key not in grid]
        if missing:
            raise ValueError(f"Grid {grid_index} does not contain: {missing}")

        # GridSearchCV stores each candidate with the keys from its source grid.
        grid_rows = [
            (row_index, row)
            for row_index, row in enumerate(rows)
            if set(row) == set(grid)
            and all(row[key] in grid[key] for key in grid)
        ]
        if not grid_rows:
            raise ValueError(f"No results matched parameter grid index {grid_index}.")

        row_indices, selected_rows = zip(*grid_rows)
        values, tick_labels = [], []
        for key in keys:
            candidates = list(grid[key])
            row_values = [row[key] for row in selected_rows]
            is_numeric = np.issubdtype(np.asarray(candidates).dtype, np.number)

            if is_numeric:
                values.append(np.asarray(row_values, dtype=float))
                tick_labels.append(None)
            else:
                categories = list(dict.fromkeys(candidates))
                values.append(
                    np.asarray([categories.index(value) for value in row_values])
                )
                tick_labels.append([str(value) for value in categories])

        scores = all_scores[list(row_indices)]
        best_local_index = int(np.argmax(scores))
        plotted_scores.extend(scores)
        subplot_best_params.append(selected_rows[best_local_index])

        points = axis.scatter(
            *values,
            c=scores,
            cmap=MSE_CMAP,
            vmin=score_min,
            vmax=score_max,
            edgecolors=COLORS["text"],
        )
        axis.scatter(
            *(value[best_local_index] for value in values),
            color=COLORS["nn"],
            marker="*",
            s=180,
            edgecolors=COLORS["text"],
            label=f"Best CV score: {values[best_local_index]:0.4f}",
        )

        axis.set_xlabel(keys[0].split("__")[-1])
        axis.set_ylabel(keys[1].split("__")[-1])
        axis.set_zlabel(keys[2].split("__")[-1])
        for plot_axis, labels in zip(
            (axis.xaxis, axis.yaxis, axis.zaxis), tick_labels
        ):
            if labels is not None:
                plot_axis.set_ticks(np.arange(len(labels)))
                plot_axis.set_ticklabels(labels)
        axis.set_title(f"Grid {grid_index + 1}: CV score")
        axis.legend()
        fig.colorbar(points, ax=axis, label="Mean CV score")

    fig.tight_layout()
    return {
        "cv_score": np.asarray(plotted_scores),
        "best_params": search.best_params_,
        "subplot_best_params": subplot_best_params,
        "figure": fig,
        "axes": axes,
    }

def plot_architecture_accuracy(search, penalty_key="neuralnetclassifier__penalty_type"):
    """Plot mean CV accuracy by hidden-layer architecture and learning rate.

    If ``penalty_key`` is provided, create a separate heatmap for each
    penalty value. Set it to ``None`` when the search has no penalty parameter.

    Returns the figure and axes.
    
    LLM assisted
    """
    
    arch_key = "neuralnetclassifier__layer_output_sizes"
    gamma_key = "neuralnetclassifier__gamma"
    results = search.cv_results_
    candidates = results["params"]

    # Preserve the order used in the search grid.
    architectures = list(dict.fromkeys(
        tuple(params[arch_key][:-1]) for params in candidates
    ))
    gammas = list(dict.fromkeys(params[gamma_key] for params in candidates))
    penalties = (
        list(dict.fromkeys(params[penalty_key] for params in candidates))
        if penalty_key is not None
        else [None]
    )

    arch_index = {arch: i for i, arch in enumerate(architectures)}
    gamma_index = {gamma: i for i, gamma in enumerate(gammas)}
    scores = results["mean_test_score"]

    fig, axes = plt.subplots(
        1, len(penalties), figsize=(5 * len(penalties), 5), squeeze=False
    )
    axes = axes.ravel()

    for ax, penalty in zip(axes, penalties):
        values = np.full((len(architectures), len(gammas)), np.nan)

        for params, score in zip(candidates, scores):
            if penalty_key is not None and params[penalty_key] != penalty:
                continue
            arch = tuple(params[arch_key][:-1])
            values[arch_index[arch], gamma_index[params[gamma_key]]] = score

        labels = [
            "No hidden layer" if not arch else "–".join(map(str, arch))
            for arch in architectures
        ]

        sea.heatmap(
            values,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap=MSE_CMAP,
            vmin=0,
            vmax=1,
            xticklabels=[f"{gamma:g}" for gamma in gammas],
            yticklabels=labels,
            cbar_kws={"label": "Mean CV accuracy"},
        )
        ax.set_title(f"Penalty: {penalty}" if penalty is not None else "No penalty")
        ax.set_xlabel("Learning rate (gamma)")
        ax.set_ylabel("Hidden-layer widths")

    fig.tight_layout()
    return fig, axes