from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random as python_random

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from load_dataset import train_dataset, validation_dataset, test_dataset


image_size = (224, 224)
num_classes = 5
seed = 68

np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


project_path = Path(__file__).resolve().parent
figures_path = project_path / "figures"
models_path = project_path / "models"

figures_path.mkdir(exist_ok=True)
models_path.mkdir(exist_ok=True)


def build_transfer_model():
    # Base model
    base_net = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3)
    )

    base_net.trainable = False

    inputs = keras.Input((image_size[0], image_size[1], 3))  # (224, 224, 3)

    x = inputs
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_net(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    net = keras.Model(inputs, outputs, name="mobilenetv2_transfer_learning")

    return net, base_net


def compile_model(net, learning_rate):
    net.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        metrics=["accuracy"]
    )


def show_learning_curves(history, file_name):
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "valid"])

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "valid"])

    plt.tight_layout()
    plt.savefig(figures_path / f"{file_name}_training_curves.png", dpi=300)
    plt.show()


def evaluate_model(net, title, file_name):
    # Evaluation
    score = net.evaluate(test_dataset)

    print()
    print(title)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Inference
    y_pred_probs = net.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_test_onehot = np.concatenate([y for _, y in test_dataset], axis=0)
    y_test = np.argmax(y_test_onehot, axis=1)

    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=test_dataset.class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=test_dataset.class_names
    )

    disp.plot(xticks_rotation=25)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figures_path / f"{file_name}_confusion_matrix.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    print("Khalil - MobileNetV2 transfer learning")
    print("Classes:", train_dataset.class_names)

    net, base_net = build_transfer_model()
    net.summary()

    # Frozen model
    compile_model(net, learning_rate=0.001)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = net.fit(
        train_dataset,
        epochs=10,
        validation_data=validation_dataset,
        callbacks=[early_stopping]
    )

    net.save(models_path / "khalil_mobilenetv2_before_finetuning.keras")

    show_learning_curves(history, "khalil_frozen_before_finetuning")

    evaluate_model(
        net,
        "MobileNetV2 frozen model",
        "khalil_frozen_before_finetuning"
    )

    # Fine tuning
    base_net.trainable = True

    for layer in base_net.layers[:-20]:
        layer.trainable = False

    for layer in base_net.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(net, learning_rate=0.00001)

    net.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = net.fit(
        train_dataset,
        epochs=10,
        validation_data=validation_dataset,
        callbacks=[early_stopping]
    )

    net.save(models_path / "khalil_mobilenetv2_finetuned.keras")

    show_learning_curves(history, "khalil_finetuned")

    evaluate_model(
        net,
        "MobileNetV2 fine-tuned model",
        "khalil_finetuned"
    )