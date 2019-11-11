import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from pathlib import Path
import logging
import sys
import os
import re
sys.path.insert(0, '/src')
from src.autoencoder import Autoencoder
from src.cnn_classifier import CNN_Classifier
from src.img_utils import read_images_in_dir, transform_images, price_filter, image_to_imagedatagen, price_filter_novel, read_novel_image
from src.image_segmentation import get_biggest_two_bounding

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
)

"""
jewelry_recommender.py  
Linh Chau
"""
## Hyperparameters
img_height = 100
img_width = 100
img_channel = 3 #RGB

price_limit = 100

image_path = 'data/earrings_full_training/780112102.jpg'
image_file = re.split('[./]', image_path)[-2]

classifer_model = 'models/jewelry_final_cnn_classifer.pkl'

output_dir = f'input_images/{image_file}'
os.makedirs(output_dir, exist_ok=True)

####### Classify jewelry type
# Segment Input Image
logging.info(f"Reading and Processing Images from {image_path}")
get_biggest_two_bounding(image_path, output_dir)
new_images = image_to_imagedatagen('input_images')

#Load in pretrained CNN Classifier
jewelry_classifier =  pickle.load(open(classifer_model, 'rb'))

# Get the probabilites and identify jewelry type
logging.info(f"Predicting Jewelry Type")
probabilities = jewelry_classifier.predict(new_images)


# ### Training set differs with type of jewelry
if np.average(probabilities) > 0.5:  
    train_dir = 'data/training_full/segmented_earrings'
    encoder_model = 'models/jewelry_cnn_encoder_earrings_test.pkl'
    knn_model = 'models/jewelry_knn_earrings_test.pkl'
else:
    train_dir = 'data/training_full/necklaces'
    encoder_model = 'models/jewelry_cnn_autoencoder_necklaces_test.pkl'
    knn_model = 'models/jewelry_knn_necklaces_test.pkl'


### Decrease Features with Autoencoding 
logging.info(f"Extracting Visual Features")
encoder =  pickle.load(open(encoder_model, 'rb'))

new_image = read_novel_image(image_path, img_height, img_width)
norm_image = np.array(new_image).astype('float32') / 255.

encoded_image = encoder.predict(norm_image.reshape(1,100,100,3))
encoded_image_flat = encoded_image.reshape((-1, np.prod(encoded_image.shape[1:])))


### KNN model to identify closet neighbors
logging.info(f"Finding Neighbors")
knn =  pickle.load(open(knn_model, 'rb'))

# Predict KNN for a test image
distances, indices = knn.kneighbors(encoded_image_flat.reshape(1, -1))

# filters neighbors based off of prices
logging.info('Price Filtering')
product_data = pd.read_csv('data/product_metadata.csv') # training product metadata
train_ids = pd.read_csv('data/training_files.csv', header=None) # training data ids, should match order of images
train_ids =list(train_ids[0])

filtered_products, product_prices, product_urls, product_images = price_filter_novel(indices[0], 
                                                                                        train_ids, 
                                                                                        product_data,
                                                                                        price_limit,)

# Plot input 
plt.imshow(read_novel_image(image_path, 256, 256))

# n_neighbors of the test image
fig= plt.figure(figsize=(20, 20))
columns = 2
rows = 2
for i, idx in zip(range(0, columns*rows), filtered_products):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(read_novel_image(product_images[i], 100, 100))
    plt.title(f'Price: {product_prices[i]}\n URL: {product_urls[i]}')
plt.show()

