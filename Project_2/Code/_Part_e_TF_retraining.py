import tensorflow_datasets as tfds
import tensorflow as tf
import joblib

from _Part_e_ import retrain_with_tensorflow

# Taken from keras example: https://www.tensorflow.org/datasets/keras_example
"""Step 1: Create your input pipeline"""
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

"""Build a training pipeline"""
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

"""Build an evaluation pipeline"""
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


"""Retraining models with tensorflow based on best models"""
model_biggrid_penalty = joblib.load("models/best_model_biggrid_penalties.joblib")
model_biggrid_penalty_params = model_biggrid_penalty.best_estimator_

best_tf_model, best_tf_history = retrain_with_tensorflow(
    model_biggrid_penalty_params, "best_model_tf", ds_train, ds_test, ds_info
)


bestCheapestModel = joblib.load("models/bestCheapest_model_biggrid_nopenalties.joblib")
cheapest_tf_model, cheapest_tf_history = retrain_with_tensorflow(
    bestCheapestModel, "best_cheapest_model_tf", ds_train, ds_test, ds_info
)



