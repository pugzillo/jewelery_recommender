# Jewelry Recommender

by Linh Chau (https://www.linkedin.com/in/linhchau/)


Jewelry is an important part of an outfit. However, compared to other wardrobe pieces, jewelry can be prohibitably expensive and difficult to shop for. I'm building a recommendation system based off an image classifier to help shoppers find jewelry pieces, specifically earrings and necklaces, that are similar in image to pieces they can input.

First, I created an image classifier with a convolutional neural network to identify what type of jewelry the input image is. Then I use an autoencoder convolutional neural network to reduce the input image to a specific set of features. The reduced features are used as input into a knn to identify jewelry in the training set that are similar visually and are in a specific price range. 

# Data
Scraped images and metadata for earrings (N = 9401) and necklaces (N = 6937) from https://www.shopstyle.com/. 

# CNN Classifier

![alt text](https://github.com/pugzillo/jewelery_recommender/images/CNN_classifier_model_loss.pdf "Log Loss for CNN Autoencoder")


# CNN Autoencoder


![alt text](https://github.com/pugzillo/jewelery_recommender/images/CNN_autoencoder_model_loss.png "Log Loss for CNN Autoencoder")


# Building a Recommender with KNN

## Distance Metric
Customers enter a specific image and the cosine similarity will be calculated to find images of jewelry that are similar to the input that fall below a certain price.

# Conclusions 


# Future Steps
1. The CNN can be further tuned to identify other types of jewelry (ie. rings).
2. Using transfer learning can potentially used to detect type of jewelry.
3. Adding features concerning non visual aspects of the jewelry pieces can be added (ie. brand).