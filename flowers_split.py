from pathlib import Path
import shutil
import random as python_random


original_path = Path("data/flower_photos")
split_path = Path("data/flowers_split")

seed = 68
python_random.seed(seed)

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15


if split_path.exists():
    shutil.rmtree(split_path)


class_names = [
    folder.name for folder in original_path.iterdir()
    if folder.is_dir()
]


for class_name in class_names:
    class_path = original_path / class_name

    image_names = [
        image.name for image in class_path.iterdir()
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    python_random.shuffle(image_names)

    n_images = len(image_names)
    n_train = int(n_images * train_ratio)
    n_validation = int(n_images * validation_ratio)

    train_images = image_names[:n_train]
    validation_images = image_names[n_train:n_train + n_validation]
    test_images = image_names[n_train + n_validation:]

    split_images = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    for split_name, split_image_names in split_images.items():
        destination_path = split_path / split_name / class_name
        destination_path.mkdir(parents=True, exist_ok=True)

        for image_name in split_image_names:
            source = class_path / image_name
            destination = destination_path / image_name

            shutil.copy2(source, destination)

    print(
        class_name,
        "Train:", len(train_images),
        "Validation:", len(validation_images),
        "Test:", len(test_images)
    )