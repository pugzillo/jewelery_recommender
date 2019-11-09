import numpy as np
import imutils
import cv2
import re
from os import listdir
from os.path import isfile, join
import os
from pathlib import Path
import errno
import multiprocessing


"""
image_segmentation.py

Takes in images of jewelry (or whatever) and returns images of the top two bounding boxes (by area)

Linh Chau
"""

def get_biggest_two_bounding(path, output_dir):
    img = cv2.imread(path)
    image_id = re.split('[./]', path)[-2]
    print(image_id)
    
    # blur and grayscale before thresholding
    blur = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(src = blur, 
                            ksize = (1,1), 
                            sigmaX = 0)
    
    # perform inverse binary thresholding 
    (t, maskLayer) = cv2.threshold(src = blur, 
                                    thresh = 0, 
                                    maxval = 255, 
                                    type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # make a mask suitable for color images
    mask = cv2.merge(mv = [maskLayer, maskLayer, maskLayer])
    
    # use the mask to select the "interesting" part of the image
    sel = cv2.bitwise_and(src1 = img, src2 = mask)
    
    contours,hierarchy = cv2.findContours(maskLayer,
                                          cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE
                                         )
    # trying to get the top 2 bounding boxes
    cont_dimen = dict()

    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        cont_dimen[area] = list([x,y,w,h])
    
    if not os.path.isdir(output_dir) :
        os.mkdir(output_dir)  # make sure the directory exists

    for i, box in zip(range(1,3),sorted(cont_dimen)[-2:]):
        print('Entering forloop')
        x,y,w,h = cont_dimen[box]
        roi=img[y:y+h+100,x:x+w+100]

        file_name = f'{output_dir}/{image_id}_{i}.jpg'
        if not cv2.imwrite(file_name, roi):
            raise Exception(f'could not write image for {filename}')
        print('end for forloop')


datasets = {
    'testing':'/Users/linhchau/Desktop/galvanize/jewelery_recommender/data/testing/earrings',
    'validation':'/Users/linhchau/Desktop/galvanize/jewelery_recommender/data/validation/earrings',
    'training':'/Users/linhchau/Desktop/galvanize/jewelery_recommender/data/training/earrings'
}

# do segmentation for all images in the following directories
for dataset, PATH in datasets.items():
    file_list = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    output_dir = Path(f'data/{dataset}/segmented_earrings')
    # print(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for file in file_list:
        get_biggest_two_bounding(f'{PATH}/{file}', output_dir)
