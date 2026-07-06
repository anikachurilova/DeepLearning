import random
import os

import shutil

original_path = "data/flower_photos/"
split_path = "data/flowers_split/"

seed = 68

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

random.seed(seed)

# Clean previous split to avoid duplicated or old files
if os.path.exists(split_path):
    shutil.rmtree(split_path)

class_names = sorted(os.listdir(original_path))

# import pdb;
# pdb.set_trace()

for class_name in class_names:
    class_folder = os.path.join(original_path, class_name)

    image_names = os.listdir(class_folder)
    image_names = [
        name for name in image_names
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # import pdb;
    # pdb.set_trace()

    random.shuffle(image_names)

    train_count = int(len(image_names) * train_ratio)
    validation_count = int(len(image_names) * validation_ratio)

    train_images = image_names[:train_count]
    validation_images = image_names[train_count:train_count + validation_count]
    test_images = image_names[train_count + validation_count:]

    split_data = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    for split_name, split_images in split_data.items():
        target_folder = os.path.join(split_path, split_name, class_name)
        os.makedirs(target_folder, exist_ok=True)

        for image_name in split_images:
            source_file = os.path.join(class_folder, image_name)
            target_file = os.path.join(target_folder, image_name)

            shutil.copy2(source_file, target_file)

    print(class_name)
    print("Train:", len(train_images))
    print("Validation:", len(validation_images))
    print("Test:", len(test_images))