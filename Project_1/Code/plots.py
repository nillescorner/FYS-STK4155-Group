""""
This python file contains the function(s?) used throughout this project to plot relevant data.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from colors import *


def plot_heatmap_grid(data, xvals, yvals, title, cbar_label, xlabel='Number of data points n',
                      ylabel=r'Noise $\sigma$', ylog=False, log=False, cmap='plasma', ax=None):
    """
    LLM Assisted. 

    Function used to plot 2D arrays MSE and R2 score to see how they display how these parameters
    are affected by number of data points (n) and noise (sigma)
    
    The whole function has been made by Claude.
    Docstring has been manually written, but takes basis in Claude's old docstring which was:
        'data: 2D array, rows = sigma (y axis), cols = n (x axis)'

        Params:
        data(NDArray): 2D array of what parameter we want to analyze
        xvals (NDarray): array with the number of datapoints
        yvals (NDArray): array with the noise values
        title (str): title of the heatmap
        cbar_label (str): title for the colorbar
        log (bool): linear og logarithmic axis
        cmap (str): choose color map

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
        
    norm = LogNorm() if log else None
    im = ax.pcolormesh(xvals, yvals, data, cmap=cmap, norm=norm, shading='nearest')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    return ax

def plot_theta(thetas, x_values, xlabel='Polynomial Degree',
               title='Coefficients', max_coeffs=None, intercept=True,
               log=False, linthresh=1e-1, xlog=False, ax=None):
    """
    LLM Assisted.
    
    Plots each coefficient theta_j as a function of x_values
    (polynomial degree, lambda, etc.).

    thetas     : list of arrays, one theta per x value (different lengths allowed)
    x_values   : degrees or lambdas, same length as thetas
    xlabel     : label for the x-axis
    max_coeffs : plot only the first max_coeffs coefficients (None = all)
    intercept  : True if theta[0] is the intercept (labels start at theta_0)
    log        : symlog y-scale
    xlog       : log x-scale (use for lambdas)
    ax         : existing axis to draw on (None = new figure)
    """
    max_len = max(len(np.ravel(t)) for t in thetas)
    theta_matrix = np.full((len(thetas), max_len), np.nan)
    for i, theta in enumerate(thetas):
        theta = np.ravel(theta)
        theta_matrix[i, :len(theta)] = theta

    n_plot = max_len if max_coeffs is None else min(max_coeffs, max_len)
    start = 0 if intercept else 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure


    if n_plot <= len(THETA_COLORS):
        colors = THETA_COLORS[:n_plot]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_plot))

    #colors = plt.cm.viridis(np.linspace(0, 1, n_plot))   #one distinct color per coefficient
    for j in range(n_plot):
        ax.plot(x_values, theta_matrix[:, j], color=colors[j], label=rf'$\theta_{{{j + start}}}$')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\theta_j$')
    ax.set_title(title)
    if xlog:
        ax.set_xscale('log')
    else:
        ax.set_xticks(x_values)
    if log:
        ax.set_yscale('symlog', linthresh=linthresh)
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    return fig, ax

def plot_cost_heatmap(cost_grid, param_values, param_label, title, fname=None,
                       cmap='plasma', y_tick_fmt="{:.3g}", relative_to=None,
                       relative_fmt=r"{:.2f}$\,\gamma_{{\max}}$"):
    """Draws a (parameter value) x (epoch) heatmap of cost, log-colored since cost
    typically spans several orders of magnitude as it decays.
 
    Some hyperparameter values (e.g. a learning rate right at or above the
    theoretical stability limit) can make SGD diverge. This shows up either as
    inf/nan cost, or as a finite but astronomically large cost (e.g. 1e250)
    just before it overflows -- both break LogNorm/its tick locator if used
    directly as vmax. So the color range is instead capped at a fixed multiple
    of the cost at epoch 0 (same for every row, since all runs start from
    theta=0): anything at or below that is "still in the game", anything above
    it (finite or not) is "diverged" and gets clipped to the cap for display.
    """
    n_epochs = cost_grid.shape[1] - 1
 
    baseline = np.nanmax(cost_grid[:, 0])  # cost at epoch 0, before any update
    cap = 100.0 * baseline                  # anything past this counts as diverged
    cost_grid_plot = np.where(np.isfinite(cost_grid), cost_grid, cap)
    cost_grid_plot = np.clip(cost_grid_plot, None, cap)
 
    finite_plot = np.isfinite(cost_grid_plot)
    vmin = max(cost_grid_plot[finite_plot].min(), 1e-12)
    vmax = cap
 
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        cost_grid_plot,
        aspect='auto',
        origin='lower',
        extent=[0, n_epochs, 0, len(param_values)],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
 
    # Label a readable subset of rows with their actual parameter value -- or,
    # when relative_to is given (e.g. gamma_max_OLS), as a fraction of that
    # reference value instead of the raw number, since "0.24 gamma_max" is more
    # meaningful here than the raw learning rate on its own.
    n_ticks = min(10, len(param_values))
    tick_idx = np.linspace(0, len(param_values) - 1, n_ticks).astype(int)
    ax.set_yticks(tick_idx + 0.5)
    if relative_to is not None:
        ax.set_yticklabels([relative_fmt.format(param_values[i] / relative_to) for i in tick_idx])
    else:
        ax.set_yticklabels([y_tick_fmt.format(param_values[i]) for i in tick_idx])
 
    ax.set_xlabel("Epoch")
    ax.set_ylabel(param_label)
    ax.set_title(title)
 
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cost (log scale)")
 
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    return fig, ax