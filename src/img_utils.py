import cv2
import numpy as np
import os
from tqdm import tqdm
import skimage.io


# Reads in images in a directory; expects only imgs in a directory
def read_images_in_dir(datadir, img_height, img_width)):
    images = []
    for i in tqdm(os.listdir(datadir)):
        path = os.path.join(datadir, i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_height, img_width))
        images.append([np.array(img)])
    return images # returns list of resized images


def transform_images(img_list):
    return img_list.astype('float32')/255.