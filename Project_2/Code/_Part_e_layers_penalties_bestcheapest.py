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
in_range = dataframe.loc[dataframe["mean_fit_time"].between(400, 600)]
best_row = in_range.loc[in_range["mean_test_score"].idxmax()]

best_model_400_600 = clone(model_biggrid_penalty.estimator)
best_model_400_600.set_params(**best_row["params"])
best_model_400_600.fit(x_train, y_train)

directory_2 = f"models/bestCheapest_model_biggrid_nopenalties.joblib"
joblib.dump(best_model_400_600, directory_2)

