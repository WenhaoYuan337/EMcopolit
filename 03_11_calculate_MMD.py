import numpy as np
import cv2
from sklearn.metrics import pairwise_kernels


def resize_image(image_path, size=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image Not Found: {image_path}")
    resized_image = cv2.resize(image, size)
    return resized_image


def standardize_image(image):
    mean = image.mean()
    std = image.std()
    return (image - mean) / std


def compute_kernel_mmd(X, Y, kernel='rbf', gamma=None):
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def load_and_preprocess_image(image_path, size=(512, 512)):
    image = resize_image(image_path, size)
    image = standardize_image(image)
    return image.flatten().reshape(1, -1)


def print_image_stats(image, name):
    print(f"{name} First five pixel value: {image.flatten()[:5]}")
    print(f"{name} Pixel value range: {image.min()} - {image.max()}")


real_image_path = 'your path'
generated_image_path = r'your path'

real_image = load_and_preprocess_image(real_image_path)
generated_image = load_and_preprocess_image(generated_image_path)
print_image_stats(real_image, "real img")
print_image_stats(generated_image, "generated img")

print(f"real_img_shape: {real_image.shape}")
print(f"generated_img_shape: {generated_image.shape}")

gammas = [0.1, 0.01, 0.001, 0.0001]
for gamma in gammas:
    mmd_value = compute_kernel_mmd(real_image, generated_image, kernel='rbf', gamma=gamma)
    print(f'Kernel MMD value: {mmd_value}')

real_image_subset = real_image[:, :1000]
generated_image_subset = generated_image[:, :1000]
for gamma in gammas:
    mmd_value_subset = compute_kernel_mmd(real_image_subset, generated_image_subset, kernel='rbf', gamma=gamma)
    print(f'Subset Kernel MMD value: {mmd_value_subset}')
