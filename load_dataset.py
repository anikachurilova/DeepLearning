from pathlib import Path

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import random as python_random


from matplotlib import pyplot as plt


# import pdb;
# pdb.set_trace()
seed = 68
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

image_size = (224, 224)
batch_size = 32



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


# Class distribution
def count_images(folder_path, class_names):
    counts = []

    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)

        image_names = os.listdir(class_path)
        image_names = [
            name for name in image_names
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        counts.append(len(image_names))

    return counts


train_amount = count_images("data/flowers_split/train", train_dataset.class_names)
validation_amount = count_images("data/flowers_split/validation", train_dataset.class_names)
test_amount = count_images("data/flowers_split/test", train_dataset.class_names)

print("\nClass distribution:")
print(f"{'Class':<12}{'Train':>8}{'Validation':>14}{'Test':>8}{'Total':>8}")

for i, class_name in enumerate(train_dataset.class_names):
    total = train_amount[i] + validation_amount[i] + test_amount[i]

    print(
        f"{class_name:<12}"
        f"{train_amount[i]:>8}"
        f"{validation_amount[i]:>14}"
        f"{test_amount[i]:>8}"
        f"{total:>8}"
    )

print()
print(f"{'Total images:':<20}{sum(train_amount) + sum(validation_amount) + sum(test_amount)}")
print(f"{'Total train:':<20}{sum(train_amount)}")
print(f"{'Total validation:':<20}{sum(validation_amount)}")
print(f"{'Total test:':<20}{sum(test_amount)}")


# Flowers distribution plot
total_amounts = []

for i in range(len(train_dataset.class_names)):
    total = train_amount[i] + validation_amount[i] + test_amount[i]
    total_amounts.append(total)

print("\nOverall class distribution:")
print(f"{'Class':<12}{'Total':>8}{'Percentage':>14}")

total_images = sum(total_amounts)

for i, class_name in enumerate(train_dataset.class_names):
    percentage = total_amounts[i] / total_images * 100

    print(
        f"{class_name:<12}"
        f"{total_amounts[i]:>8}"
        f"{percentage:>13.2f}%"
    )


# Bar plot for total class distribution
plt.figure(figsize=(8, 5))

plt.bar(train_dataset.class_names, total_amounts)

plt.xlabel("Flower class")
plt.ylabel("Number of images")
plt.title("Overall class distribution in the Flowers dataset")
plt.xticks(rotation=25)

plt.tight_layout()
project_path = Path(__file__).resolve().parent
plt.savefig(project_path/"figures/class_distribution.png", dpi=300)
plt.show()


# Images examples
plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        class_index = np.argmax(labels[i].numpy())
        plt.title(train_dataset.class_names[class_index])

        plt.axis("off")

plt.tight_layout()
plt.savefig("figures/example_flower_images.png", dpi=300)
plt.show()