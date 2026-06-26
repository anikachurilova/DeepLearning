import tensorflow as tf

import numpy as np
import random as python_random

from tensorflow import keras


# import pdb;
# pdb.set_trace()

base_path = "data/flowers_split/"

np.random.seed(0)
python_random.seed(0)
tf.random.set_seed(0)

image_size = (224, 224)
batch_size = 32
seed = 68


# Training
train_dataset = keras.utils.image_dataset_from_directory(
    "data/flowers_split/train",
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True,
    seed=seed,
)

# Validation
validation_dataset = keras.utils.image_dataset_from_directory(
    "data/flowers_split/validation",
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True,
    seed=seed,
)

# Test
test_dataset = keras.utils.image_dataset_from_directory(
    "data/flowers_split/test",
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False
)

# Some info about the dataset
print("Flowers class names:", train_dataset.class_names)
print("Amuont of classes:", len(train_dataset.class_names))

# Check shapes
for images, labels in train_dataset.take(1):
    print("Images shape:", images.shape)
    print("Images dtype:", images.dtype)
    print("Labels shape:", labels.shape)
    print("Labels dtype:", labels.dtype)
