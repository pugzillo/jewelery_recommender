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

# Get images from the directories
train_data = read_images_in_dir(train_dir)
test_data = read_images_in_dir(test_dir)

# Normalize the image pixels to 0-1
trans_train_data = transform_images(train_data)
trans_test_data = transform_images(test_data)


### Setting up CNN Autoencoder Model


