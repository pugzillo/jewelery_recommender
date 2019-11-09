import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import CSVLogger
from livelossplot import PlotLossesKeras
from keras.models import Sequential

"""
cnn_classifier.py  
Linh Chau
"""

class CNN_Classifier():
    def __init__(self):
        self.classifier = None
        self.log_file = None

    def fit(self, X_train, X_test, batch_size = 32, n_epochs = 10):
        self.log_file = f'{X_train}_cnnclassifier_log.csv'
        self.classifier.fit(X_train,
                            validation_data = X_test,
                            steps_per_epoch = len(X_train.filenames) // batch_size,
                            validation_steps =  len(X_test.filenames) // batch_size,
                            epochs = n_epochs, 
                            callbacks=[PlotLossesKeras(), CSVLogger(self.log_file,
                                        append=False,
                                        separator=";")], 
                            # verbose = 1
        )

    def predict(self, X):
        return self.classifier.predict(X)

    def compile(self, opt = 'adam', loss_func = 'binary_crossentropy'):
        self.classifier.compile(optimizer = opt, loss = loss_func, metrics=['accuracy'])
        print(self.classifier.summary())
    
    def set_architecture(self, img_width, img_height, img_channel):
        """
        Architecture inspired by ...
        """
        input_shape = (img_width, img_height, img_channel)

        self.classifier = Sequential()

        self.classifier.add(Conv2D(32, (3, 3), input_shape=input_shape))
        self.classifier.add(Activation('relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        self.classifier.add(Conv2D(64, (3, 3)))
        self.classifier.add(Activation('relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        self.classifier.add(Conv2D(128, (3, 3)))
        self.classifier.add(Activation('relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        self.classifier.add(Flatten())
        self.classifier.add(Dense(128))
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(1))
        self.classifier.add(Activation('sigmoid'))

    def metrics(self, X):
        return self.classifier.evaluate_generator(X)


