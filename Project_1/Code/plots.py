""""
This python file contains the function(s?) used throughout this project to plot relevant data.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_heatmap_grid(data, xvals, yvals, title, cbar_label, xlabel='Number of data points n', 
                      ylabel=r'Noise $\sigma$', ylog=False, log=False, cmap='plasma'):
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

    fig, ax = plt.subplots(figsize=(7, 5))
    norm = LogNorm() if log else None
    im = ax.pcolormesh(xvals, yvals, data, cmap=cmap, norm=norm, shading='nearest')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
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

    colors = plt.cm.viridis(np.linspace(0, 1, n_plot))   # one distinct color per coefficient
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