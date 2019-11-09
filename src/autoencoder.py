import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam

"""
autoencoder.py  
Linh Chau
"""

class Autoencoder():
    def __init__(self):
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    def fit(self, X_train, X_test, batch_size = 256, n_epochs = 500):
        self.autoencoder.fit(X_train, X_train,
                            validation_data = (X_test, X_test),
                            batch_size = batch_size,
                            epochs = n_epochs, 
                            # verbose = 1
        )

    def encoder_predict(self, X):
        return self.encoder.predict(X)

    def compile_autoencoder(self, opt = 'adam', loss_func = 'binary_crossentropy'):
        self.autoencoder.compile(optimizer = opt, loss = loss_func)
    
    def set_architecture(self, img_width, img_height, img_channel):
        """
        Architecture inspired by CNN autoencoder tutorial: 
        https://blog.keras.io/building-autoencoders-in-keras.html
        """

        input_img = Input(shape=(img_width, img_height, img_channel))  # adapt this if using `channels_first` image data format
        n_hidden_1, n_hidden_2, n_hidden_3 = 16, 8, 8
        convkernel = (3,3) # convolution kernal
        poolkernel = (2,2) #pooling kernal

        x = Conv2D(n_hidden_1, convkernel, activation='relu', padding='same')(input_img)
        x = MaxPooling2D(poolkernel, padding='same')(x)
        x = Conv2D(n_hidden_2, convkernel, activation='relu', padding='same')(x)
        x = MaxPooling2D(poolkernel, padding='same')(x)
        x = Conv2D(n_hidden_3, convkernel, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(poolkernel, padding='same')(x)

        x = Conv2D(n_hidden_3, convkernel, activation='relu', padding='same')(encoded)
        x = UpSampling2D(poolkernel)(x)
        x = Conv2D(n_hidden_2, convkernel, activation='relu', padding='same')(x)
        x =  UpSampling2D(poolkernel)(x)
        x =  Conv2D(n_hidden_1, convkernel, activation='relu')(x)
        x = UpSampling2D(poolkernel)(x)
        decoded =  Conv2D(img_channel, convkernel, activation='sigmoid', padding='same')(x)

        # autoencoder model 
        self.autoencoder = Model(input_img, decoded)

        # encoder model
        self.encoder = Model(input_img, encoded)

        # decoder model
        self.decoder = Model(input_img, encoded)

        print(self.autoencoder.summary())
        
    
 

