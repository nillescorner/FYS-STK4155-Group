import matplotlib.pyplot as plt
import numpy as np

def plot_metric(x, train_data, test_data=None, titles=None, params=None, ylabel='',
                 yscale='linear', ncols=None, figsize=None, legend=True):
    """
    Plot one metric (MSE, R2, theta, ...) across panels — one panel per entry
    in train_data. Just plots what you give it, nothing else.

        Parameters:
            x (array): shared x-axis (e.g. polynomial degrees)
            train_data (list of arrays): one curve per panel
            test_data (list of arrays or None): optional second curve per panel
            titles (list of str, str template, or None):
                - list of str: one title per panel, used as-is
                - str with '{}': filled per panel using params[i] (or panel index if params is None)
                - None: no titles
            params (list or None): values to fill into a titles template, one per panel
            ylabel (str): y-axis label (put on first panel only)
            yscale (str): 'linear' or 'log'
            ncols (int or None): defaults to len(train_data)
            figsize (tuple or None): defaults based on ncols
            legend (bool): whether to show a legend (only makes sense if test_data given)
    """
    ncols = ncols or len(train_data)
    figsize = figsize or (12, 4)

    if isinstance(titles, str):
        fill = params if params is not None else range(len(train_data))
        titles = [titles.format(p) for p in fill]

    fig, ax = plt.subplots(1, ncols, sharey=True, figsize=figsize)
    ax = np.atleast_1d(ax).flatten()

    for i in range(len(train_data)):
        ax[i].plot(x, train_data[i], 'o-', label='Training data')
        if test_data is not None:
            ax[i].plot(x, test_data[i], 'o-', label='Testing data')
        if titles is not None:
            ax[i].set_title(titles[i])
        ax[i].set_xlabel('Polynomial Degree')
        ax[i].set_xticks(x)
        if yscale != 'linear':
            ax[i].set_yscale(yscale)

    ax[0].set_ylabel(ylabel)
    if legend and test_data is not None:
        ax[0].legend()

    plt.tight_layout()
    plt.show()
    return fig, ax