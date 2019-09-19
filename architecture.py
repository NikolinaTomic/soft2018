from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


class Model:
    @staticmethod
    def build_network(classes, height, width, channels):

        if K.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        return model


