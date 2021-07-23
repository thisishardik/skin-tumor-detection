import os
import gc
import json
import math
import cv2
import PIL
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import glob
import tensorflow.keras.applications.densenet as dense
from kaggle_datasets import KaggleDatasets
import seaborn as sns

def get_random_eraser(input_img, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
   # def eraser(input_img):
    img_h, img_w, img_c = input_img.shape

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))
    else:
        c = np.random.uniform(v_l, v_h)

    input_img[top:top + h, left:left + w, :] = c

    return input_img

if __name__ == "__main__":
    TRAIN = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
    IMAGE_SIZE = 1024
    n_imgs = 12
    img_filenames = os.listdir(TRAIN)[:n_imgs]
    img_filenames[:3]
    image=[]
    for file_name in img_filenames:
        img = cv2.imread(TRAIN +file_name)
        img = get_random_eraser(img)
        image.append(img)
    grid_display(image, 4, (15,15))