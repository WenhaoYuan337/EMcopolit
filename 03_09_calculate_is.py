import tensorflow as tf
import numpy as np
from scipy.stats import entropy
import os
from PIL import Image

inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')

def get_predictions(images):
    processed_images = tf.keras.applications.inception_v3.preprocess_input(images)
    preds = inception_model.predict(processed_images)

    return preds


def calculate_inception_score(images, num_splits=10):
    splits = np.array_split(images, num_splits)
    scores = []

    for images in splits:
        preds = get_predictions(images)
        p = np.mean(preds, axis=0)
        kl_scores = []
        for pred in preds:
            kl = entropy(pred, p)
            kl_scores.append(kl)

        mean_kl = np.mean(kl_scores)
        is_score = np.exp(mean_kl)
        scores.append(is_score)

    is_mean = np.mean(scores)
    is_std = np.std(scores)

    return is_mean, is_std


dataset_folder = 'your fold'
image_files = os.listdir(dataset_folder)
image_data = []

for file in image_files:
    image_path = os.path.join(dataset_folder, file)
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize((299, 299))
    image = np.array(image)
    image_data.append(image)

image_data = np.array(image_data)
is_mean, is_std = calculate_inception_score(image_data)
print("Inception Score: {:.4f} +/- {:.2f}".format(is_mean, is_std))
