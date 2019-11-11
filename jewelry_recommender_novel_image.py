import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from src.img_utils import read_images_in_dir, transform_images, price_filter
from src.autoencoder import Autoencoder
import logging
from src.cnn_classifier import CNN_Classifier
import pickle
from src.image_segmentation import get_biggest_two_bounding

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
)

"""
jewelry_recommender.py  
Linh Chau
"""
## Hyperparametets
img_height = 100
img_width = 100
img_channel = 3 #RGB

price_limit = 100

image_path = 'data/earrings_full_training/794612301.jpg'
classifer_model = 'jewelry_cnn_earrings_test.pkl'


output_dir = Path(f'input_data_{image_path}')
os.makedirs(output_dir, exist_ok=True)

####### Classify jewelry type
# Segment Input Image
logging.info(f"Reading and Processing Images from {image_path}")
img1, img2 = get_biggest_two_bounding(image_path, output_dir)
new_images = image_to_imagedatagen(output_dir)

#Load in pretrained CNN Classifier
jewelry_classifier =  pickle.load(open(classifer_model, 'rb'))

# Get the probabilites and identify jewelry type
logging.info(f"Predicting Jewelry Type")
probabilities = model_3.predict_generator(new_images, 2)


### Training set differs with type of jewelry
if :
    train_dir = 'data/training_full/segmented_earrings'
    encoder_model = ''
    knn_model = ''
else:
    train_dir = 'data/training_full/necklaces'
    encoder_model = ''
    knn_model = ''


train_data, train_ids = read_images_in_dir(train_dir, img_height, img_width)

### Decrease Features with Autoencoding 
logging.info(f"Extracting Visual Features")
encoder =  pickle.load(open(encoder_model, 'rb'))

new_image = read_images_in_dir(image_path, img_height, img_width)
norm_image = np.array(new_image).astype('float32') / 255.

encoded_image = encoder.predict(norm_image)
encoded_image_flat = encoded_image.reshape((-1, np.prod(encoded_image.shape[1:])))


### KNN model to identify closet neighbors
logging.info(f"Finding Neighbors")
knn =  pickle.load(open(knn_model, 'rb'))

# Predict KNN for a test image
distances, indices = knn.kneighbors(encoded_train_flat.reshape(1, -1))

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
    plt.title(f'Price: {product_prices[i-1]}\n URL: {product_urls[i-1]}')
plt.show()

