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

    def fit(self, X_train, X_test, batch_size = 32, n_epochs = 50):
        self.autoencoder.fit(X_train, X_train,
                            validation_data = (X_test, X_test),
                            batch_size = batch_size,
                            epochs = n_epochs, 
                            verbose = 1
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
        hidden_1, hidden_2 = 16, 8
        con_kernal = (3,3) # convolution kernal
        pool_kernal = (2,2) #pooling kernal

        x = Conv2D(hidden_1, con_kernal, activation='relu', padding='same')(input_img)
        x = MaxPooling2D(pool_kernal, padding='same')(x)
        x = Conv2D(hidden_2, con_kernal, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_kernal, padding='same')(x)
        x = Conv2D(hidden_2, con_kernal, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pool_kernal, padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional if input (width = 28, height = 28, channel= 3)

        x = Conv2D(hidden_2, con_kernal, activation='relu', padding='same')(encoded)
        x = UpSampling2D(pool_kernal, padding='same')(x)
        x = Conv2D(hidden_2, con_kernal, activation='relu', padding='same')(x)
        x = UpSampling2D(pool_kernal, padding='same')(x)
        x = Conv2D(hidden_1, con_kernal, activation='relu')(x)
        x = UpSampling2D(pool_kernal, padding='same')(x)
        decoded = Conv2D(img_channel, con_kernal, activation='sigmoid', padding='same')(x)

        # autoencoder model 
        self.autoencoder = Model(input_img, decoded)

        # encoder model
        self.encoder = Model(input_img, encoded)

        # decoder model
        self.decoder = Model(input_img, encoded)

        print(self.autoencoder.summary())
        
    
 

