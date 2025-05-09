import random

import numpy as np
import pandas as pd
from img2vec_pytorch import Img2Vec
from PIL import Image

import image_utils

cat = image_utils.get_images_from_dir("dataset/processed_images/cat")
horse = image_utils.get_images_from_dir("dataset/processed_images/horse")
bird = image_utils.get_images_from_dir("dataset/processed_images/bird")
random_cat = random.sample(cat, 10)
random_horse = random.sample(horse, 10)
random_bird = random.sample(bird, 10)

concat_paths = random_cat + random_horse + random_bird
concat_images = [image_utils.load_image(path) for path in concat_paths]

print(concat_images)


# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=False)
embeddings = img2vec.get_vec(concat_images)

print(embeddings.shape)

df = pd.DataFrame(embeddings)
df.insert(0, "filepaths", concat_paths)
df.to_csv("embeddings/embeddings.csv", index=False)
