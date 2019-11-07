import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from src.img_utils import read_images_in_dir, transform_images
from src.autoencoder import Autoencoder

"""
jewelry_recommender.py  
Linh Chau
"""

### Processing Images 
# Directories with Data
train_dir = 'data/training/earrings'
test_dir = 'data/testing/earrings'

img_height = 256
img_width = 256
img_channel = 3 #RGB

# Get images from the directories
train_data = read_images_in_dir(train_dir, img_height, img_width)
test_data = read_images_in_dir(test_dir, img_height, img_width)

# Normalize the image pixels to 0-1
trans_train_data = transform_images(train_data)
trans_test_data = transform_images(test_data)


### Setting up CNN Autoencoder Model
autoencoder = Autoencoder()
autoencoder.set_architecture(img_width, img_height, img_channel)
autoencoder.compile_autoencoder()
autoencoder.fit(train_data, test_data)

## Encoded layer for both the train and test data
encoded_train = autoencoder.predict(train_data)
encoded_test = autoencoder.predict(test_data)

## flatten the encoded img, so they are the shape (#imgs, height*width*channels of output of encoder)- input for KNN
encoded_train_flat = encoded_train.reshape((-1, np.prod(encoded_train.shape[1:])))
encoded_test_flat = encoded_test.reshape((-1, np.prod(encoded_test.shape[1:])))


##KNN to find items similar to a test image

# Making Training set with the encoded training layer
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(encoded_train_flat)

test_img = 7

# Predict KNN for a test image
distances, indices = knn.kneighbors(encoded_train_flat[test_img].reshape(1, -1))

# print n_neighbors of the test image
fig= plt.figure(figsize=(8, 8))
columns = 3
rows = 2
for i, idx in zip(range(1, columns*rows +1), np.nditer(indices)):
    fig.add_subplot(rows, columns, i)
    plt.imshow(tr_img_data[idx].reshape(28, 28, 3))
plt.show()