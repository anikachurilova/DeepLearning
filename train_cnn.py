from pathlib import Path
import random as python_random
import csv

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from load_dataset import (
    train_dataset,
    validation_dataset,
    test_dataset,
    seed
)

from cnn_from_scratch import create_compiled_cnn



np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

models_path = Path(__file__).resolve().parent/"models"
figures_path = Path(__file__).resolve().parent / "figures"
tables_path = Path(__file__).resolve().parent / "tables"

model = create_compiled_cnn()

history = model.fit(
    train_dataset,
    epochs=40,
    validation_data=validation_dataset
)

# Save trained model and history
model.save(models_path / "cnn_from_scratch.keras")
history_file = tables_path / "cnn_from_scratch_history.csv"

with open(history_file, "w", newline="") as file:
    fieldnames = ["epoch"] + list(history.history.keys())
    dictWriter = csv.DictWriter(file, fieldnames=fieldnames)

    dictWriter.writeheader()

    for i in range(len(history.history["loss"])):
        row = {"epoch": i + 1}

        for key in history.history.keys():
            row[key] = history.history[key][i]

        dictWriter.writerow(row)


# Plot accuracy curve
plt.figure(figsize=(10, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "validation"])
plt.title("Loss result for CNN from scratch")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylim([0.5, 1.0])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"])
plt.title("Accuracy result for CNN from scratch")

plt.tight_layout()
plt.savefig(figures_path / "cnn_from_scratch_training_curves.png", dpi=300)
plt.show()


# Evaluate on test set
score = model.evaluate(test_dataset)

print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Prediction
y_pred_probs = model.predict(test_dataset)
print("Prediction shape(probability distribution):", y_pred_probs.shape)

# Selection of the highest-probability class for each test sample
y_pred = np.argmax(y_pred_probs, axis=1)
print('Prediction shape(argmax):', y_pred.shape)

# Ground truth classes
y_test_onehot = np.concatenate([y for _, y in test_dataset], axis=0)
y_test = np.argmax(y_test_onehot, axis=1)

# Classification report
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=test_dataset.class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
plt.imshow(cm, cmap=plt.cm.Blues)

plt.xlabel("Predicted class")
plt.ylabel("Ground truth class")
plt.title("Confusion matrix for CNN from scratch")

plt.xticks(
    ticks=np.arange(len(test_dataset.class_names)),
    labels=test_dataset.class_names,
    rotation=45
)

plt.yticks(
    ticks=np.arange(len(test_dataset.class_names)),
    labels=test_dataset.class_names
)

plt.colorbar()
plt.tight_layout()

plt.savefig(figures_path / "cnn_from_scratch_confusion_matrix.png", dpi=300)
plt.show()