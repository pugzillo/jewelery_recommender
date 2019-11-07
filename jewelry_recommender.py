import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from src.img_utils import read_images_in_dir, transform_images
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

### Processing Images 
# Directories with Data
train_dir = 'data/training/earrings'
test_dir = 'data/testing/earrings'

img_height = 100
img_width = 100
img_channel = 3 #RGB

# Get images from the directories
logging.info(f"Reading and Processing Images from {train_dir}")
train_data = read_images_in_dir(train_dir, img_height, img_width)

logging.info(f"Reading and Processing Images from {test_dir}")
test_data = read_images_in_dir(test_dir, img_height, img_width)

# Normalize the image pixels to 0-1
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
pickle_name = 'jewelry_cnn_autoencoder_test.sav'
pickle.dump(knn, open(pickle_name, 'wb'))



##KNN to find items similar to a test image

# Making Training set with the encoded training layer
logging.info('Finding Similar items to input')
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(encoded_train_flat)

test_img = 7
plt.imshow(test_data[test_img])

# Predict KNN for a test image
distances, indices = knn.kneighbors(encoded_train_flat[test_img].reshape(1, -1))

# print n_neighbors of the test image
fig= plt.figure(figsize=(8, 8))
columns = 3
rows = 2
for i, idx in zip(range(1, columns*rows +1), np.nditer(indices)):
    fig.add_subplot(rows, columns, i)
    plt.imshow(train_data[idx].reshape(256, 256, 3))
plt.show()



