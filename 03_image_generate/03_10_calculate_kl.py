import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_grayscale_histogram(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    histogram /= histogram.sum()

    return histogram


def calculate_kl_divergence(hist1, hist2):
    kl_divergence = cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)

    return kl_divergence


dataset1_folder = 'dataset1 path'
dataset2_folder = 'dataset2 path'

dataset1_files = [file for file in os.listdir(dataset1_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]
dataset2_files = [file for file in os.listdir(dataset2_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]
assert len(dataset1_files) == len(dataset2_files), "The number of images in dataset1 and dataset2 must be the same."
kl_divergences = []

for image1_file, image2_file in zip(dataset1_files, dataset2_files):
    image1_path = os.path.join(dataset1_folder, image1_file)
    image2_path = os.path.join(dataset2_folder, image2_file)

    histogram1 = generate_grayscale_histogram(image1_path)
    histogram2 = generate_grayscale_histogram(image2_path)

    kl_divergence = calculate_kl_divergence(histogram1, histogram2)
    kl_divergences.append(kl_divergence)

average_kl_divergence = np.mean(kl_divergences)
print("Average KL Divergence:", average_kl_divergence)
