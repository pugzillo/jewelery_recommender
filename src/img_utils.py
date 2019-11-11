import cv2
import numpy as np
import os
from tqdm import tqdm
import skimage.io
from keras.preprocessing.image import ImageDataGenerator

"""
img_utils.py 

Functions useful when reading in images for cnn autoencoder >> knn for image similarity retrival and cnn classifier

Linh Chau
"""

# Reads in images in a directory; expects only imgs in a directory
def read_images_in_dir(datadir, img_height, img_width):
    images = []
    image_ids = []
    for i in tqdm(os.listdir(datadir)):
        path = os.path.join(datadir, i)
        image_ids.append(os.path.splitext(os.path.basename(path))[0])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_height, img_width))
        images.append(np.array(img))
    return images, image_ids 

# Reads in a single image
def read_novel_image(imagedir, img_height, img_width):
    img = cv2.imread(imagedir, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_height, img_width))
    return np.array(img)

def transform_images(img_list):
    return np.array(img_list).astype('float32')/255.

# takes in directory of images to create imagedatagenerator objects for cnn
def image_to_imagedatagen(dir, img_width = 100, img_height = 100, batch_size = 32, class_mode=None, shuffle=False):
    # rescales images
    datagen = ImageDataGenerator(rescale=1./255)
    #Creates generator object
    generator = datagen.flow_from_directory(
        dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        color_mode = 'rgb',
        class_mode = class_mode,
        shuffle = shuffle
    )
    return(generator)

# filters the price of the neighbooring images
def price_filter(neighbors, product_ids, product_data, price_limit):
    filtered_products = []
    product_prices = []
    product_urls = []
    for neighbor in neighbors:
        product_id = product_ids[neighbor]
        price = float(product_data[product_data['id'] == int(product_id)]['price'])
        url = str(product_data[product_data['id'] == int(product_id)]['clickUrl'])
        if price < price_limit:
            filtered_products.append(neighbor)
            product_prices.append(price)
            product_urls.append(url)
    return filtered_products, product_prices, product_urls 