"""
This python file contains the generalized functions used throughout this project.

The Runge function for our data, function for setting up the design matrix, 
making the x and y data, and scaling for data where it also can be split into
test and training data.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.datasets import load_digits
import json




#Path in order to save figures in Project_2/figs, LLM Assisted 
FIG_DIR = Path(__file__).resolve().parent.parent / "figs"   # Project_2/figs
FIG_DIR.mkdir(exist_ok=True)

#Path in order to save results learning rate results so they stays consistent, LLM Assisted
RESULTS_DIR = Path(__file__).resolve().parent / "intermediary_results"
RESULTS_DIR.mkdir(exist_ok=True)

import argparse # to toggle saving figs or displaying them

class ArgsHelper:
    """Parses command line flags once and offers convenience methods.
    LLM assisted lvl 2
    
    Usage:
        #pass args in command line
        example : python3 _Part_b.py --save-fig

        Passable arguments:
            --save-figs
            --load-bestlrs
    
    """

    def __init__(self, fig_dir=FIG_DIR):
        self.fig_dir = Path(fig_dir)

        parser = argparse.ArgumentParser()
        parser.add_argument("--save-figs", action="store_true",
                            help="save figures to disk instead of showing them")
        parser.add_argument("--load-bestlrs", action="store_true",
                            help="load the previously found best learning rates")

        # parse_known_args tolerates flags this class doesn't know about
        self.args, _ = parser.parse_known_args()

    @property
    def load_bestlrs(self):
        """True if --load-bestlrs was passed. Use in an if statement."""
        return self.args.load_bestlrs

    def plot_or_save(self, fname):
        """Save the current figure if --save-figs was passed, otherwise show it."""
        if self.args.save_figs:
            print(f"saving figs to {self.fig_dir}/")
            plt.savefig(self.fig_dir / fname)
        else:
            plt.show()
        plt.close()


#Save to best learning rates
best_lrs = RESULTS_DIR / "best_lr.json"

def save_best_lr(key, value):
    """Add or update one entry in best_lr.json, keeping the others
        LLM Assisted"""
    data = load_best_lr()
    data[key] = value
    with open(best_lrs, "w") as f:
        json.dump(data, f, indent=2)

def load_best_lr(key=None):
    """Read best_lr.json (all of it, or one entry)
    LLM Assisted"""
    data = json.loads(best_lrs.read_text()) if best_lrs.exists() else {}
    return data if key is None else data[key]


def runge(x):
    """
    Runge function given in Project 1 description

        Params:
            x (NDArray or float): input values to generate Runge function
        
        Returns:
            f(x) (NDarray or float): The Runge function evaluated at input x
    """

    return 1.0 / (1.0 + 25.0 * x**2)


class RungeData:
    """
    Noisy data from the Runge function, y = f(x) + N(0, noise^2), x uniform in [-1, 1]
    Default at n = 100, noise = 0.1, seed = 2026
 
        Params:
            n (int): Number of data points
            noise (float): Noise on the data (sigma)
            seed (int): Use the same seed to generate same numbers 
 
        Attributes:
            x (NDArray): input data x, shape: (n,)
            y (NDArray): target data, shape: (n,)
    """

    def __init__(self, n = 100, noise = 0.1, seed = 2026):
        self.n, self.noise, self.seed = n, noise, seed
        rng = np.random.default_rng(seed)
        self.x = jnp.asarray(np.sort(rng.uniform(-1, 1, n)))
        self.y = runge(self.x) + jnp.asarray(rng.normal(0, noise, n))

    @staticmethod
    def design_matrix(x, degree, intercept=True):
        """
        Creates design matrix X
        Polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)

            Params:
                x (NDArray): input data
                degree (int): degree of polynomial
                intercept (bool): if False, column of ones is left out
            
            Returns:
                X (NDArray): design matrix
        """
        start = 0 if intercept else 1
        return jnp.vstack([x**p for p in range(start, degree + 1)]).T

    def polynomial_features(self, degree, intercept=True):
        """
        Design matrix of this data set 
        """

        return self.design_matrix(self.x, degree, intercept)
 
    def nn_features(self):
        """
        Raw x as a single input feature for the neural network, shape: (n, 1)
        """

        return self.x.reshape(-1, 1)
 
    def split(self, X, test_size=0.2):
        """
        Train-test split of a feature matrix X and this data's y, using the data's seed.
 
            Params:
                X (NDArray): features
                test_size (float): fraction of the data used for testing
 
            Returns:
                X_train, X_test, y_train, y_test (NDArray)
        """

        return train_test_split(X, self.y, test_size=test_size, random_state=self.seed)


class Scaler:
    """
    Normalizes a design matrix X and centers an array y, with the means and stds
    computed from the data it is fitted on 
 
    Based on p116 fit with intercept from lecturebook.
 
        Usage:
            scaler = Scaler().fit(X_train, y_train)
            X_train_s, y_train_s = scaler.transform(X_train, y_train)
            X_test_s, y_test_s = scaler.transform(X_test, y_test)
            y_pred = scaler.inverse_transform_y(model_output)   
    """
 
    def fit(self, X, y=None):
        """
        Computes the column means and stds of X, and the mean of y
 
            Params:
                X (NDArray): Design matrix
                y (NDArray or None): Target data
 
            Returns:
                self (Scaler): so that Scaler().fit(X, y) can be chained
        """

        self.X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self.X_std = jnp.where(X_std == 0, 1, X_std)   # leave constant columns alone
        self.y_mean = y.mean() if y is not None else 0.0
        return self
 
    def transform_X(self, X):
        """
        (X - X_mean) / X_std, with the fitted mean and std
        """
        return (X - self.X_mean) / self.X_std
 
    def transform_y(self, y):
        """
        y - y_mean, with the fitted mean
        """
        return y - self.y_mean
 
    def inverse_transform_y(self, y_scaled):
        """
        y_scaled + y_mean: brings predictions back to the original scale
        """
        return y_scaled + self.y_mean
 
    def transform(self, X, y):
        """
        Scales both X and y, 
        Returns: X_norm, y_centered
        """
        return self.transform_X(X), self.transform_y(y)
 
    def fit_transform(self, X, y):
        """
        Fits on X, y and scales them, 
        Returns: X_norm, y_centered
        """
        return self.fit(X, y).transform(X, y)



def scaling(X, y, seed=2026, split_data=False):
    """
    If split_data=True it returns X_train_norm, X_test_norm, y_train_centered, y_test_centered
    (test data scaled with the training mean/std), otherwise X_norm, y_centered
    """

    if split_data:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        scaler = Scaler().fit(X_train, y_train)
        return (scaler.transform_X(X_train), scaler.transform_X(X_test), scaler.transform_y(y_train), scaler.transform_y(y_test))
    
    return Scaler().fit_transform(X, y)


class MnistImportAndScale:
    """
    Imports and scales MNIST dataset. 
    Upon first run per user with fulldataset = True, dataset will be downloaded 
    Uses either full or reduced dataset. Both are split with train-test being 5-1.

    Usage: 
    mnist_data = MnistImportAndScale()
    fig, ax = plt.subplots()
    mnist_data.display_image(idx = 2,ax = ax)
    """

    def __init__(self, fulldataset = True, seed=2026):
        self.seed = seed

        if fulldataset:
            #using full dataset from TensorFlow
            self.mnist_file  = tf.keras.datasets.mnist.load_data()
            (x_train, self.y_train), (x_test, self.y_test) = self.mnist_file 

            self.x_train = x_train.reshape(x_train.shape[0], -1) / 255  #/255 bcs there are 255 pixel vals
            self.x_test = x_test.reshape(x_test.shape[0], -1) /255

            self.x_train_img = x_train / 255
            self.x_test_img = x_test / 255

        else:
            #Using smaller dataset from Scikit-learn for testing
            self.mnist_file = load_digits()
            x = self.mnist_file['data']
            y = self.mnist_file['target']
            x_img = self.mnist_file['images'] #convenient form for plotting later
            
            x_train, x_test, self.y_train, self.y_test, x_train_img, x_test_img = train_test_split(x, y, x_img,  test_size=5/6, random_state=seed)

            self.x_train = x_train / 255  #/255 bcs there are 255 pixel vals
            self.x_test = x_test /255

            self.x_train_img = x_train_img / 255
            self.x_test_img = x_test_img / 255

        self.imgxydim = self.x_train[1].size #dimension of pixelated image


    def set_digit(self, digit):
        "Sets y_train and y_test to digit. To train on cases like 4 or not 4, aka binary cases"
        self.y_train = (self.y_train == digit).astype(float).reshape(-1, 1)
        self.y_test = (self.y_test == digit).astype(float).reshape(-1, 1)
        
    def return_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def display_image(self, idx, ax, display_train_img = True):
        """Displays image of test or train of specific idx
                Params:
                    idx : index of data to display
                    ax : plt.ax object to plot on 
                    display_train_img(Boolean) : If True, uses train split data. Else uses test split data
        """
        if display_train_img:
            ax.matshow(self.x_train_img[idx], cmap="gray")
        else: 
            ax.matshow(self.x_test_img[idx], cmap="gray")

    def kfold_splits(self, n_splits = 5, shuffle = True, seed = 2026):
        """Splits in k-folds and returns indices to use in cross-validation loops
        
            Indexing:
                [(train_indices, validation_indices)*k] = mnistobj.kfold_splits()
        
            Usage: 
                splits =  mnistobj.kfold_splits()
                train_idx, val_idx = splits[0]  # first fold
                X_fold_train, y_fold_train = x_train[train_idx], y_train[train_idx]
                X_fold_val, y_fold_val = x_train[val_idx], y_train[val_idx]
        """
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state = seed)
        self.splits =  list(kf.split(self.x_train))
        return self.splits

