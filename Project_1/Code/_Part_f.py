
from optimizer_methods import *
from general_functions import *
from colors import *
import numpy as np
import matplotlib.pyplot as plt

gammas = np.logspace(-3, 0, 15)

degree = 5
lam = 1e-2

#making data
x_raw, y_raw = make_data()
X_raw = design_matrix(x_raw, degree, intercept = False)

#centering data
X, y = scaling(X_raw, y_raw, split_data = False)

FUNCT_RUNS = (('plain', gammas, PLAIN ),
            ('momentum', gammas, MOMENTUM),
            ('adagrad', gammas, ADAGRAD),
            ('rmsprop', gammas, RMSPROP),
            ('adam', gammas, ADAM))

num_iters=30000
excess5, eigs5, gamma5 = funct_comparison(X = X,y = y, FUNCT_RUNS=FUNCT_RUNS, num_iters=num_iters)
tolerances = (1e-4, 1e-8)
iterations = {}
for method, gammas_, color in FUNCT_RUNS:
    e_ = excess5[method]
    its_per_gamma = {}
    for gamma in gammas_:
        e = e_[gamma] #accesssing the excess per method per learning rate gamma
        its_per_gamma[gamma] = {tol: (int(np.argmax(e < tol)) if (e < tol).any() else None) for tol in tolerances}
        print(f"{method:8s} (gamma = {gamma5 if gamma is None else gamma:.3f}): to 1e-4: {its_per_gamma[gamma][1e-4]}, to 1e-8: {its_per_gamma[gamma][1e-8]}, "
            f"after {num_iters}: {max(e[-1], 0):.1e}")
    iterations[method] = its_per_gamma


#plotting iterations agains gammas for all methods on log axes

fig, axs = plt.subplots(1, len(tolerances),  figsize=(6.4*len(tolerances), 4.0))
# fig, axs = plt.subplots(len(tolerances),1,  figsize=(6.4, 4.0*len(tolerances)))
for ax, tolerance in zip(axs,tolerances):
    for method, gammas_, col, in FUNCT_RUNS:
        its = np.array([
            np.nan if iterations[method][gamma][tolerance] is None
            else iterations[method][gamma][tolerance]
            for gamma in gammas_
        ])

        ax.loglog(gammas_, its, "o-", color=col, label=method)
    ax.axvline(2 / eigs5.max(), color="grey", ls="--", lw=1, label=r"$2/\lambda_{\max}$")
    ax.axvline(2 *(1+0.9)/ eigs5.max(), color="#854646", ls="--", lw=1, label=r"$2(1+\beta)/\lambda_{\max}$") #beta = 0.9 for momentum
    ax.set_ylabel(r"iterations to excess cost $<$" +f"{tolerance}")
    ax.set_xlabel(r"learning rate $\gamma$"); 
    ax.grid(alpha = 0.5, which = 'both')
axs[-1].set_xlabel(r"learning rate $\gamma$"); 
fig.suptitle("Needed iterations for convergence of different machine learning methods based on learning rates")
ax.legend(fontsize=8); fig.tight_layout()
plt.savefig(FIG_DIR / "Part_f_MethodsComparison_withLearningRate.png")
