import tensorflow as tf
from tensorflow import keras
import numpy as np


import random as python_random


image_size = (224, 224)
num_classes = 5
seed = 68
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


def build_cnn_from_scratch(num_classes=5):
    inputs = keras.Input((image_size[0], image_size[1], 3)) # (224, 224, 3)

    x = inputs
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(3, strides=3, padding="same")(x)

    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalMaxPooling2D()(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_from_scratch")

    return model


if __name__ == "__main__":
    model = build_cnn_from_scratch(num_classes=num_classes)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=["accuracy"]
    )

    model.summary()