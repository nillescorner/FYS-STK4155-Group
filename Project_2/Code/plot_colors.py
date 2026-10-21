
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
 


COLORS = {
    "true":        "#2B1B24",   #the Runge function: dark plum ink
    "train_data":  "#B79AA9",   #training points: muted mauve
    "test_data":   "#6E4F60",   #test points: darker mauve, square markers
    "nn":          "#D6336C",   #neural network fit: rose pink 
    "ols":         "#7A2DB0",   #OLS / Ridge / Lasso fit: plum, dashed
    "grid":        "#EFE3E8",   #light blush grid lines
    "text":        "#2B1B24",   #label color
    }

#Four categorical colors: pink, plum, amber, teal.
SERIES = ["#D6336C", "#7A2DB0", "#D46A1E", "#00908F"]

#fixed mapping
OPTIMIZER_COLORS = {
    "GD":            SERIES[0],
    "SGD":           SERIES[1],
    "SGD + RMSprop": SERIES[2],
    "SGD + Adam":    SERIES[3],
}

ARCH_COLORS = {
    "1 x 50":  SERIES[0],
    "1 x 100": SERIES[1],
    "2 x 50":  SERIES[2],
    "2 x 100": SERIES[3],
}

ACTIVATION_COLORS = {
    "Sigmoid":    SERIES[0],
    "ReLU":       SERIES[1],
    "LeakyReLU":  SERIES[2],
}

REG_COLORS = {
    "None": SERIES[0],
    "L2":   SERIES[1],
    "L1":   SERIES[2],
}


#linestyles
TRAIN_LS = "-"
VAL_LS = "--"

#colormap for heatmaps
MSE_CMAP = LinearSegmentedColormap.from_list("pink_seq", ["#FDF0F4", "#F4B6CB", "#D6336C", "#8E1F55", "#3D0F2E"])

def set_style():
    """
    Sets matplotlib defaults so every figure in the project looks the same
    """

    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=SERIES),    #Setting default colors
        "axes.edgecolor": "#8C7A84",                  #Color of axis                 
        "axes.labelcolor": COLORS["text"],              #color of label
        "axes.titlesize": 12,                           #size of text in title
        "axes.labelsize": 11,                           #size of text on axis
        "axes.spines.top": False,                       #removes top axis
        "axes.spines.right": False,                     #removes right axis
        "axes.grid": True,                              #sets grid
        "grid.color": COLORS["grid"],                   #sets color of grid
        "grid.linewidth": 0.8,                          #linewidth of grid (thinner than plot lines)
        "xtick.color": "#6E5F67",                     #color of labels on x axis                    
        "ytick.color": "#6E5F67",                     #color of labels on y axis
        "legend.frameon": False,                        #removes box around label
        "legend.fontsize": 9,                           #size of text in labels
        "lines.linewidth": 1.7,                         #default linewidth of plotted lines
        "figure.figsize": (7, 4.5),                     #default figure size
        "figure.dpi": 110,                              #resolution of figure
        "savefig.dpi": 200,                             #resolution of figure when saved
        "savefig.bbox": "tight",                        #cuts off extra white spacing when saving figure
        "font.size": 10,                                #default fontsize
    })


def plot_data_and_truth(ax, x_grid, y_true, X_train, y_train, X_test, y_test):
    """
    Draws the true function and the data points 
    """

    ax.scatter(X_train.ravel(), y_train, s=14, color=COLORS["train_data"], alpha=0.8, edgecolor="white", linewidth=0.5, label="Training data", zorder=2)
    ax.scatter(X_test.ravel(), y_test, s=16, marker="s", color=COLORS["test_data"], alpha=0.9, edgecolor="white", linewidth=0.5, label="Test data", zorder=2)
    ax.plot(x_grid, y_true, color=COLORS["true"], lw=1.5, label="Runge function", zorder=3)