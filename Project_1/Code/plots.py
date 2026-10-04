""""
This python file contains the function(s?) used throughout this project to plot relevant data.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_heatmap_grid(data, xvals, yvals, title, cbar_label, xlabel='Number of data points n',
                      ylabel=r'Noise $\sigma$', ylog=False, log=False, cmap='plasma', ax=None,
                      annotate=False, fmt='.2g', fontsize=7, best=None):
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
        annotate (bool): write the value of each cell inside the cell (best for small grids)
        fmt (str): number format for the annotations, e.g. '.2g', '.3f'
        fontsize (int): font size of the annotations
        best (str or None): 'min' marks the lowest value (MSE), 'max' marks the highest (R2),
                            None marks nothing
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
        
    norm = LogNorm() if log else None
    im = ax.pcolormesh(xvals, yvals, data, cmap=cmap, norm=norm, shading='nearest')
    im.autoscale_None()     #make sure the colour limits are set before we look up cell colours
    if ylog:
        ax.set_yscale('log')

    #Write the value in each cell, black text on light cells and white text on dark cells
    if annotate:
        for i, yv in enumerate(yvals):
            for j, xv in enumerate(xvals):
                val = data[i, j]
                if np.isnan(val):
                    continue
                r, g, b, _ = im.cmap(im.norm(val))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(xv, yv, format(val, fmt), ha='center', va='center',
                        fontsize=fontsize, color='black' if luminance > 0.5 else 'white')

    #Mark the best cell: outline the cell if values are written in it, otherwise a star
    if best is not None:
        flat_index = np.nanargmin(data) if best == 'min' else np.nanargmax(data)
        i, j = np.unravel_index(flat_index, data.shape)
        label = f'Best = {format(data[i, j], fmt)}'

        if annotate:
            #cell size taken from the spacing to neighbouring grid points
            w = np.gradient(np.asarray(xvals, dtype=float))[j]
            h = np.gradient(np.asarray(yvals, dtype=float))[i]
            rect = plt.Rectangle((xvals[j] - w / 2, yvals[i] - h / 2), w, h, fill=False,
                                 edgecolor='white', linewidth=2.5, label=label)
            ax.add_patch(rect)
        else:
            ax.plot(xvals[j], yvals[i], marker='*', markersize=15, color='white',
                    markeredgecolor='black', linestyle='none', label=label)

        #legend placed above the plot, to the right of the title, so it covers no cells
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), fontsize=8,
                  frameon=False, borderaxespad=0.2)

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