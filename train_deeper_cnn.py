from pathlib import Path
import random as python_random
import tensorflow as tf
from tensorflow import keras

import csv

import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from load_dataset import (
    train_dataset,
    validation_dataset,
    test_dataset,
    image_size,
    seed
)


np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

models_path = Path(__file__).resolve().parent / "models"
figures_path = Path(__file__).resolve().parent / "figures"
tables_path = Path(__file__).resolve().parent / "tables"

# Deeper CNN from scratch (with data augmentation and early stopping, but no dropout)
def create_deeper_cnn():
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))

    x = inputs

    # Data augmentation
    x = keras.layers.RandomFlip("horizontal", seed=seed)(x)
    x = keras.layers.RandomRotation(0.1, seed=seed)(x)
    x = keras.layers.RandomZoom(0.1, seed=seed)(x)

    # Deeper CNN structure
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(3, strides=3, padding="same")(x)

    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(3, strides=3, padding="same")(x)

    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalMaxPooling2D()(x)

    outputs = keras.layers.Dense(len(train_dataset.class_names), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="deeper_cnn")

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=["accuracy"]
    )

    return model


model = create_deeper_cnn()

model.summary()

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True  # it goes back to the epoch where validation loss was the best
)

# Train
history = model.fit(
    train_dataset,
    epochs=80,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

# Save trained model and history
model.save(models_path / "deeper_cnn.keras")
history_file = tables_path / "deeper_cnn_history.csv"

with open(history_file, "w", newline="") as file:
    fieldnames = ["epoch"] + list(history.history.keys())
    dictWriter = csv.DictWriter(file, fieldnames=fieldnames)

    dictWriter.writeheader()

    for i in range(len(history.history["loss"])):
        row = {"epoch": i + 1}

        for key in history.history.keys():
            row[key] = history.history[key][i]

        dictWriter.writerow(row)


# Plot accuracy and loss curves
plt.figure(figsize=(10, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "validation"])
plt.title("Loss result for deeper CNN")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylim([0.5, 1.0])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"])
plt.title("Accuracy result for deeper CNN")

plt.tight_layout()
plt.savefig(figures_path / "deeper_cnn_training_curves.png", dpi=300)
plt.show()

# Evaluate on test set
score = model.evaluate(test_dataset)

print()
print("Deeper CNN test loss:", score[0])
print("Deeper CNN test accuracy:", score[1])

# Prediction
y_pred_probs = model.predict(test_dataset)
print("Prediction shape(probability distribution):", y_pred_probs.shape)

# Selection of the highest-probability class for each test sample
y_pred = np.argmax(y_pred_probs, axis=1)
print("Prediction shape(argmax):", y_pred.shape)

# Ground truth classes
y_test_onehot = np.concatenate([y for _, y in test_dataset], axis=0)
y_test = np.argmax(y_test_onehot, axis=1)

# Classification report
print("Classification report:")
print(classification_report( y_test, y_pred, target_names=test_dataset.class_names,))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
plt.imshow(cm, cmap=plt.cm.Blues)

plt.xlabel("Predicted class")
plt.ylabel("Ground truth class")
plt.title("Confusion matrix for deeper CNN")

plt.xticks(
    ticks=np.arange(len(train_dataset.class_names)),
    labels=train_dataset.class_names,
    rotation=45
)

plt.yticks(
    ticks=np.arange(len(train_dataset.class_names)),
    labels=train_dataset.class_names
)

plt.colorbar()
plt.tight_layout()

plt.savefig(figures_path / "deeper_cnn_confusion_matrix.png", dpi=300)
plt.show()