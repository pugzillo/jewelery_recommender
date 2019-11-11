# Hack your Look: An Image-based Jewelry Recommender

by Linh Chau (https://www.linkedin.com/in/linhchau/)


Jewelry is an important part of an outfit. However, compared to other wardrobe pieces, jewelry can be prohibitably expensive and difficult to shop for. I'm building a recommendation system based off an image classifier to help shoppers find jewelry pieces, specifically earrings and necklaces, that are similar in image to pieces they can input.

First, I created an image classifier with a convolutional neural network to identify what type of jewelry the input image is. Then I use an autoencoder convolutional neural network to reduce the input image to a specific set of features. The reduced features are used as input into a knn to identify jewelry in the training set that are similar visually and are in a specific price range. 


## Data
Scraped images and metadata for earrings (N = 9401) and necklaces (N = 6937) from [ShopStyle](https://www.shopstyle.com/ "Shop Style"). 90% of earrings and necklaces were used for the training set (N = 23,155) and 10% of each type were used for the testing set. 

## Building a Jewelry Classifier with a CNN Classifier
A convolutional neural network (CNN) was used to identify if an image contained a necklace or earring. For training of the CNN, segmentation was performed on earring images given that they come in pairs. Images were converted to RGB and resized to 100x100.

The model was ran for 15 epochs.

![alt text](https://github.com/pugzillo/jewelery_recommender/blob/master/images/CNN_classifier_model_loss.pdf "Log Loss for CNN Classifier")


## CNN Autoencoder
A convolutional neural network autoencoder was used to reduce the dimensions of each training and input image. 

![alt text](https://github.com/pugzillo/jewelery_recommender/blob/master/images/CNN_autoencoder_model_loss.png "Log Loss for CNN Autoencoder")


## Building a Recommender with KNN

K-nearst neighbors was used to identify images, with reduced features from the cnn autoencoder, that are similar to the input image. The **cosine similarity** will be calculated to find images of jewelry that are similar to the input that fall below a certain price.

## Running pipeline for a novel image
python3 jewelry_recommender_novel_image.py 

## Conclusions and Future Work
Even though I had a small set of training images, I was about to build a recommender using a KNN algorithm with features created by the dimension reduction capabilites of a CNN autoencoder. This allows users to input pictures of jewelry that they are interested in and get similar pieces in a certain price range with links for purchase. 

In the future, I'd like to focus on the following:
1. The CNN can be further tuned with more training images or with other types of jewelry (ie. rings).
2. Using transfer learning can potentially used to detect type of jewelry.
3. Adding features concerning non visual aspects of the jewelry pieces can be added (ie. brand).


