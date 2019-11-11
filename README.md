# Jewelry Recommender

by Linh Chau (https://www.linkedin.com/in/linhchau/)


Jewelry is an important part of an outfit. However, compared to other wardrobe pieces, jewelry can be prohibitably expensive and difficult to shop for. I'm building a recommendation system based off an image classifier to help shoppers find jewelry pieces, specifically earrings and necklaces, that are similar in image to pieces they can input.

First, I created an image classifier with a convolutional neural network to identify what type of jewelry the input image is. Then I use an autoencoder convolutional neural network to reduce the input image to a specific set of features. The reduced features are used as input into a knn to identify jewelry in the training set that are similar visually and are in a specific price range. 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents** 

- [Jewelry Recommender:](#jewelry-recommender)
  - [Data:](#data)
  - [CNN Classifier:](#CNN-Classifier)
  - [CNN Autoencoder:](#CNN-Autoencoder)
  - [Building a Recommender with KNN:](#Building-a-Recommender-with-KNN)
  - [Conclusions:](#conclusions)
  - [Future Work:](#future-work)
  - [References:](#references)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Data
Scraped images and metadata for earrings (N = 9401) and necklaces (N = 6937) from https://www.shopstyle.com/. 

## CNN Classifier

![alt text](https://github.com/pugzillo/jewelery_recommender/blob/master/images/CNN_classifier_model_loss.pdf "Log Loss for CNN Classifier")


## CNN Autoencoder
images/CNN_autoencoder_diagram.png

![alt text](https://github.com/pugzillo/jewelery_recommender/blob/master/images/CNN_autoencoder_model_loss.png "Log Loss for CNN Autoencoder")


## Building a Recommender with KNN

### Distance Metric
Customers enter a specific image and the cosine similarity will be calculated to find images of jewelry that are similar to the input that fall below a certain price.

## Conclusions 


# Future Work
1. The CNN can be further tuned to identify other types of jewelry (ie. rings).
2. Using transfer learning can potentially used to detect type of jewelry.
3. Adding features concerning non visual aspects of the jewelry pieces can be added (ie. brand).

# References