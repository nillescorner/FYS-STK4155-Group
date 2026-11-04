from general_functions import MnistImportAndScale
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sea

from _Part_a import ReLU, LeakyReLU, Sigmoid, Identity, Softmax, MulticlassCrossEntropy
from optimizer import Adam, train, SGD
from neural_network import NeuralNetwork, NeuralNetRegressor, NeuralNetClassifier
from general_functions import RungeData, Scaler, FIG_DIR, save_best_lr, load_best_lr, ArgsHelper


from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve)

import joblib
import os

from plot_colors import *

n_jobs_set = int(joblib.cpu_count()*0.8)
print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))
print("Joblib CPUs:", joblib.cpu_count(), f", will use {n_jobs_set} CPUs")

set_style()


mnist_data = MnistImportAndScale()
x_train, x_test, y_train, y_test = mnist_data.return_data()
cv_splits = mnist_data.kfold_splits(n_splits=5)

""" Checking gradient before running training  """
neural_network  = NeuralNetClassifier(cost=MulticlassCrossEntropy, 
                                        seed=2026, check_gradients=True)
neural_network  = NeuralNetClassifier(cost=MulticlassCrossEntropy, 
                                        seed=2026, eval_every=5)
net = make_pipeline(neural_network)

n_classes = len(np.unique(y_train))

# Each entry lists hidden-layer widths; the output layer is added below.
hidden_architectures = [
    [],
    [50],
    [50, 50],
    [100, 50],
    [100, 50, 25],
]

param_grid = []

for hidden_sizes in hidden_architectures:
    layer_sizes = hidden_sizes + [n_classes]
    activations = [ReLU for _ in hidden_sizes] + [Softmax]

    param_grid.append({
        "neuralnetclassifier__layer_output_sizes": [layer_sizes],
        "neuralnetclassifier__activation_functions": [activations],
        "neuralnetclassifier__gamma": [1e-4, 3e-4, 1e-3],
        "neuralnetclassifier__lmbd": [0, 1e-5, 1e-4],
        "neuralnetclassifier__penalty_type": ['L1', 'L2'],
    })


directory = f"models/best_model_biggrid_penalties.joblib"
print("directory:", directory)

search = GridSearchCV(
    net,
    param_grid,
    cv=cv_splits,
    scoring="accuracy",
    return_train_score=True,
    n_jobs=n_jobs_set,
    refit=True,
)
fit_NN = search.fit(x_train, y_train)


#saving model. pretty easy yay NOTE: opening train model only works if enviroment is the same (supposedly)
joblib.dump(search, directory)


""" No penalties (no lambdas)"""

param_grid = []

for hidden_sizes in hidden_architectures:
    layer_sizes = hidden_sizes + [n_classes]
    activations = [ReLU for _ in hidden_sizes] + [Softmax]

    param_grid.append({
        "neuralnetclassifier__layer_output_sizes": [layer_sizes],
        "neuralnetclassifier__activation_functions": [activations],
        "neuralnetclassifier__gamma": [1e-4, 3e-4, 1e-3],
    })

directory_2 = f"models/best_model_biggrid_nopenalties.joblib"
print("directory:", directory_2)

search_ = GridSearchCV(
    net,
    param_grid,
    cv=cv_splits,
    scoring="accuracy",
    return_train_score=True,
    n_jobs=n_jobs_set,
    refit=True,
)
fit_NN_ = search_.fit(x_train, y_train)


#saving model. pretty easy yay NOTE: opening train model only works if enviroment is the same (supposedly)
joblib.dump(search_, directory_2)
