import pandas as pd
import joblib
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from general_functions import MnistImportAndScale


mnist_data = MnistImportAndScale()
x_train, x_test, y_train, y_test = mnist_data.return_data()
cv_splits = mnist_data.kfold_splits(n_splits=5)

model_biggrid_penalty = joblib.load("models/best_model_biggrid_penalties.joblib")
best_idx = model_biggrid_penalty.best_index_
results = model_biggrid_penalty.cv_results_

dataframe = pd.DataFrame(results)

best_LG_idx = dataframe[0:18]["mean_test_score"].idxmax()
best_LG_score = dataframe[0:18].loc[best_LG_idx, "mean_test_score"]
best_LG_row = dataframe[0:18].loc[best_LG_idx, "params"]
print(best_LG_row)

lg_model = clone(model_biggrid_penalty.estimator)
lg_model.set_params(**best_LG_row)
lg_model.fit(x_train, y_train)

directory = f"models/logistic_regression_model.joblib"
joblib.dump(lg_model, directory)