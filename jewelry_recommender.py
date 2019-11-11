import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from src.img_utils import read_images_in_dir, transform_images, price_filter
from src.autoencoder import Autoencoder
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
)

"""
jewelry_recommender.py  
Linh Chau
"""

# Hyperparameter
# Directories with Data
train_dir = 'data/training_full/segmented_earrings'
test_dir = 'data/testing_full/segmented_earrings'

img_height = 100
img_width = 100
img_channel = 3 #RGB

price_limit = 100


### Processing Images 
# Get images from the directories
logging.info(f"Reading and Processing Images from {train_dir}")
train_data, train_ids = read_images_in_dir(train_dir, img_height, img_width)

logging.info(f"Reading and Processing Images from {test_dir}")
test_data, test_ids = read_images_in_dir(test_dir, img_height, img_width)

# Normalize the image pixels to 0-1
logging.info(f"Normalizing the images!")
trans_train_data = transform_images(train_data)
trans_test_data = transform_images(test_data)



### Setting up CNN Autoencoder Model
logging.info('Setting up the Autoencoder')
autoencoder = Autoencoder()
autoencoder.set_architecture(img_width, img_height, img_channel)
autoencoder.compile_autoencoder()
autoencoder.fit(trans_train_data, trans_test_data)

## Encoded layer for both the train and test data
logging.info('Putting Images through the encoded layer')
encoded_train = autoencoder.encoder_predict(trans_train_data)
encoded_test = autoencoder.encoder_predict(trans_test_data)

## flatten the encoded img, so they are the shape (#imgs, height*width*channels of output of encoder)- input for KNN
encoded_train_flat = encoded_train.reshape((-1, np.prod(encoded_train.shape[1:])))
encoded_test_flat = encoded_test.reshape((-1, np.prod(encoded_test.shape[1:])))

## Save model in pickle
pickle_names = ['jewelry_cnn_encoder_earrings_test.pkl', 'jewelry_cnn_autoencoder_earrings_test.pkl', 'jewelry_cnn_earrings_test.pkl']
autoencoder_models = [autoencoder.encoder, autoencoder.autoencoder, autoencoder.decoder]

for model, pickles in zip(autoencoder_models, pickle_names):
    pickle.dump(model, open(pickles, 'wb'))


##KNN to find items similar to a test image

# Making Training set with the encoded training layer
logging.info('Finding Similar items to input')
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(encoded_train_flat)

pickle_name = 'jewelry_knn_earrings_test.pkl'
pickle.dump(knn, open(pickle_name, 'wb'))

test_img = 7
plt.imshow(test_data[test_img])

# Predict KNN for a test image
distances, indices = knn.kneighbors(encoded_train_flat[test_img].reshape(1, -1))

# filters neighbors based off of prices
logging.info('Price Filtering')
product_data = pd.read_csv('data/product_metadata.csv') # training product metadata

filtered_products, product_prices, product_urls = price_filter(indices[0], 
                                                   train_ids, 
                                                   product_data, 
                                                   price_limit)

# n_neighbors of the test image
fig= plt.figure(figsize=(20, 20))
columns = 3
rows = 2
for i, idx in zip(range(1, columns*rows +1), filtered_products):
    fig.add_subplot(rows, columns, i)
    plt.imshow(train_data[idx].reshape(img_height, img_width, img_channel))
    plt.title(f'Price: {product_prices[i-1]}')
plt.show()






