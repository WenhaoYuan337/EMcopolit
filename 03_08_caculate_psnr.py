import os
import cv2
import numpy as np
from tqdm import tqdm


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    return psnr


def resize_image(img):
    resized_img = cv2.resize(img, (512, 512))
    return resized_img


folder1 = r'path1'
folder2 = r'path2'
psnr_values = []

for filename in tqdm(os.listdir(folder1)):
    img1 = cv2.imread(os.path.join(folder1, filename))
    img2 = cv2.imread(os.path.join(folder2, filename))

    resized_img1 = resize_image(img1)
    resized_img2 = resize_image(img2)

    psnr = calculate_psnr(resized_img1, resized_img2)
    psnr_values.append(psnr)

average_psnr = np.mean(psnr_values)
print("Average PSNR:", average_psnr)