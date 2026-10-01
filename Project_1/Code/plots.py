""""
This python file contains the function(s?) used throughout this project to plot relevant data.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_heatmap_grid(data, xvals, yvals, title, cbar_label, log=False, cmap='plasma'):
    """
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

    ax.set_xlabel('Number of data points n')
    ax.set_ylabel(r'Noise $\sigma$')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    return ax

