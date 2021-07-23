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


def onehot(image, label):
    CLASSES = 2
    return image, tf.one_hot(label, CLASSES)


def cutmix(image, label):
    DIM = 1024  # IMAGE_SIZE[0]
    CLASSES = 2

    imgs = []
    labs = []
    for j in range(AUG_BATCH):
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
        WIDTH = tf.cast(DIM * tf.math.sqrt(1-b), tf.int32)
        ya = tf.math.maximum(0, y-WIDTH//2)
        yb = tf.math.minimum(DIM, y+WIDTH//2)
        xa = tf.math.maximum(0, x-WIDTH//2)
        xb = tf.math.minimum(DIM, x+WIDTH//2)
        # MAKE CUTMIX IMAGE
        one = image[j, ya:yb, 0:xa, :]
        two = image[k, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle,
                         image[j, yb:DIM, :, :]], axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/DIM/DIM, tf.float32)
        if len(label.shape) == 1:
            lab1 = tf.one_hot(label[j], CLASSES)
            lab2 = tf.one_hot(label[k], CLASSES)
        else:
            lab1 = label[j, ]
            lab2 = label[k, ]
        labs.append((1-a)*lab1 + a*lab2)

    image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (AUG_BATCH, CLASSES))
    return image2, label2


if __name__ == "__main__":
    AUG_BATCH = 48
    row = 6
    col = 4
    row = min(row, AUG_BATCH//col)
    all_elements = training_dataset.unbatch()
    augmented_element = all_elements.repeat().batch(AUG_BATCH).map(cutmix)

    for (img, label) in augmented_element:
        plt.figure(figsize=(15, int(15*row/col)))
        for j in range(row*col):
            plt.subplot(row, col, j+1)
            plt.axis('off')
            plt.imshow(img[j, ])
        plt.show()
        break
