import numpy
import os
import cv2
import paddle
from paddle_msssim import ssim, ms_ssim
import argparse
import torch
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def calculate_fid(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(np.dot(sigma1, sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def data_list(pred_dir, GT_dir):
    generated_Dataset = []
    real_Dataset = []
    for file in os.listdir(GT_dir):
        realImg = cv2.imread(os.path.join(GT_dir, file)).astype('float32')
        realImg = cv2.resize(realImg, (299, 299))
        real_Dataset.append(realImg)
        generatedImg = cv2.imread(os.path.join(pred_dir, file)).astype('float32')
        generatedImg = cv2.resize(generatedImg, (299, 299))
        generated_Dataset.append(generatedImg)
    return generated_Dataset, real_Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='result/facades', help='results')
    parser.add_argument('--GT_dir', type=str, default='dataset/real', help='name of dataset')
    opt = parser.parse_args()

    generatedImg, realImg = data_list(opt.pred_dir, opt.GT_dir)
    dataset_size = len(generatedImg)
    print("dataset size：", dataset_size)

    images1 = torch.Tensor(generatedImg)
    images2 = torch.Tensor(realImg)
    print('shape: ', images1.shape, images2.shape)

    model = InceptionV3(include_top=False, pooling='avg')

    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    images1 = tf.cast(images1, tf.float32)
    images2 = tf.cast(images2, tf.float32)
    fid = calculate_fid(model, images1, images2)
    print('FID_average : %.3f' % (fid / dataset_size))

