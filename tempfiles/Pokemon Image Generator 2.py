import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import PIL
from PIL import Image
import time
import os
import helpfile
from glob import glob
import pickle as pkl 
import scipy.misc
import cv2
from datetime import datetime

do_preprocess = False
from_checkpoint = False

def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)

    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch

def main():
    data_dir = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\MyScrapedImages\\" # Data


if __name__=="__main__":
    main()