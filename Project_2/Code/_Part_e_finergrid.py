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

# starting with a coarser grid 
""" Checking gradient before running training  """
neural_network  = NeuralNetClassifier([50, 10], [ReLU(), Softmax()], Adam, cost=MulticlassCrossEntropy(), 
                                        seed=2026, check_gradients=True)

neural_network  = NeuralNetClassifier([50, 10], [ReLU(), Softmax()], Adam, cost=MulticlassCrossEntropy(),
                                        epochs= 150, penalty_type= 'L2',
                                        seed=2026)

net = make_pipeline(neural_network)


param_grid = {"neuralnetclassifier__gamma": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
              "neuralnetclassifier__lmbd": [0, 1e-6, 1e-5, 1e-4, 1e-3],
              "neuralnetclassifier__penalty_type": ['L2', 'L1', None]}
search = GridSearchCV(net, param_grid, cv=cv_splits, scoring="accuracy", return_train_score=True, n_jobs=n_jobs_set, refit=True) 
fit_NN = search.fit(x_train, y_train)


#saving model. pretty easy yay NOTE: opening train model only works if enviroment is the same (supposedly)
ext_name = "x".join(str(len(values)) for values in param_grid.values())
joblib.dump(search, f"models/best_model_{ext_name}.joblib")
