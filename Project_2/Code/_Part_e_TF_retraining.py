import tensorflow_datasets as tfds
import tensorflow as tf
import joblib

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

def retrain_with_tensorflow(selected_pipeline, model_name):
    """Rebuild a selected sklearn classifier in Keras, train it, and report test metrics.
    
    LLM assisted"""
    selected_net = selected_pipeline.named_steps["neuralnetclassifier"]
    params = selected_net.get_params()
    tf.keras.utils.set_random_seed(params["seed"])

    penalty_type = params["penalty_type"]
    penalty_strength = params["lmbd"]
    if penalty_type == "L2":
        regularizer = tf.keras.regularizers.L2(penalty_strength)
    elif penalty_type == "L1":
        regularizer = tf.keras.regularizers.L1(penalty_strength)
    elif penalty_type is None or penalty_strength == 0:
        regularizer = None
    else:
        raise ValueError(f"Unsupported penalty type: {penalty_type}")

    activation_map = {
        "ReLU": "relu",
        "LeakyReLU": tf.keras.layers.LeakyReLU(),
        "Sigmoid": "sigmoid",
        "Identity": None,
    }
    layer_sizes = params["layer_output_sizes"]
    activations = params["activation_functions"]
    if len(layer_sizes) != len(activations):
        raise ValueError("The selected model must specify one activation per layer.")
    if activations[-1].__name__ != "Softmax":
        raise ValueError("Expected the selected classifier to use a Softmax output.")

    regularizer_args = {"kernel_regularizer": regularizer} if regularizer is not None else {}
    layers = [
        tf.keras.layers.Input(shape=tuple(ds_info.features["image"].shape)),
        tf.keras.layers.Flatten(),
    ]
    for width, activation_class in zip(layer_sizes[:-1], activations[:-1]):
        activation_name = activation_class.__name__
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported hidden activation: {activation_name}")
        layers.append(
            tf.keras.layers.Dense(
                width,
                activation=activation_map[activation_name],
                **regularizer_args,
            )
        )
    # Keep logits as output; from_logits=True applies a numerically stable softmax loss.
    layers.append(tf.keras.layers.Dense(layer_sizes[-1], **regularizer_args))

    keras_model = tf.keras.Sequential(layers, name=model_name)
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["gamma"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    train_size = ds_info.splits["train"].num_examples
    steps_per_epoch = (train_size + params["batch_size"] - 1) // params["batch_size"]
    train_data = (
        ds_train.unbatch()
        .batch(params["batch_size"])
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    print(f"Training {model_name} with: {params}")
    history = keras_model.fit(
        train_data,
        epochs=params["epochs"],
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        shuffle=False,
    )
    test_loss, test_accuracy = keras_model.evaluate(ds_test, verbose=0)
    print(f"{model_name} test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{model_name}.keras"
    keras_model.save(save_path)
    print(f"Saved {model_name} to {save_path}")
    return keras_model, history


"""Retraining models with tensorflow based on best models"""
model_biggrid_penalty = joblib.load("models/best_model_biggrid_penalties.joblib")
model_biggrid_penalty_params = model_biggrid_penalty.best_estimator_

best_tf_model, best_tf_history = retrain_with_tensorflow(
    model_biggrid_penalty_params, "best_model_tf"
)


bestCheapestModel = joblib.load("models/bestCheapest_model_biggrid_nopenalties.joblib")
cheapest_tf_model, cheapest_tf_history = retrain_with_tensorflow(
    bestCheapestModel, "best_cheapest_model_tf"
)



