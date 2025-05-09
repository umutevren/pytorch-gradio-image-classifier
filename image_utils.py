import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _check_is_dir(path, make=False):
    if not os.path.exists(path):
        if make:
            os.makedirs(path, exist_ok=True)
            return True
        else:
            raise ValueError(f"Path does not exist: {path}")

    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")

    return True


def filter_images(list_of_files):
    valid_extensions = {".jpg", ".png", ".jpeg", ".webp"}
    return [
        file
        for file in list_of_files
        if any(file.endswith(ext) for ext in valid_extensions)
    ]


def get_images_from_dir(dir_path):
    _check_is_dir(dir_path)
    files = os.listdir(dir_path)
    image_files = filter_images(files)
    image_paths = [os.path.join(dir_path, file) for file in image_files]
    return image_paths


def read_images_from_dir(dir_path):
    _check_is_dir(dir_path)
    files = os.listdir(dir_path)
    image_files = filter_images(files)
    image_paths = [os.path.join(dir_path, file) for file in image_files]
    images = [load_image(image_path) for image_path in image_paths]
    logging.info(f"Loaded {len(images)} images from {dir_path}")
    return images


def min_resolution_filter(image, min_width, min_height):
    width, height = image.size
    return width >= min_width and height >= min_height


def max_resolution_rescale(image, max_width, max_height):
    width, height = image.size
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def center_crop(image, new_width, new_height):
    width, height = image.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    cropped_image = image.crop((left, top, right, bottom))
    logging.info(f"Center cropped image to {new_width}x{new_height}")
    return cropped_image


# def normalize_image(image):
#     np_image = np.array(image).astype(float)
#     normalized_image = (np_image - np.min(np_image)) / (
#         np.max(np_image) - np.min(np_image)
#     )
#     return Image.fromarray((normalized_image * 255).astype(np.uint8))


def resize_image(image, width, height):
    resized_image = image.thumbnail((width, height))
    return resized_image


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def save_image(image, save_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise ValueError("Input image must be a numpy array or PIL Image")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(save_path)
    logging.info(f"Saved image to {save_path}")


def plot_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def plot_images(images, cols=2):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(images, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def convert_to_jpg(image, save_path):
    image = image.convert("RGB")
    image.save(save_path)
    logging.info(f"Saved image to {save_path}")
    return image


def save_images_to_dir(images, dir_path):
    _check_is_dir(dir_path)
    for i, image in enumerate(images, 1):
        save_path = os.path.join(dir_path, f"image_{i}.jpg")
        save_image(image, save_path)
        logging.info(f"Saved image to {save_path}")
    return True
