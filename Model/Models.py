from __future__ import division, print_function

from keras.layers import Input, Conv3D, Concatenate, Flatten, Dense
from keras.layers import BatchNormalization, Activation
from keras.models import Model


def dilated_densenet(height, width, length, channels, classes, features=12, depth=4, 
                     temperature=1.0, padding='same', batchnorm=False,
                     dropout=0.0):
    x = Input(shape=(height, width, length, channels))
    inputs = x
    print(inputs)

    # initial convolution
    x = Conv3D(features, kernel_size=(5,5,5), padding=padding)(x)

    maps = [inputs]
    dilation_rate = 1
    kernel_size = (3,3,3)
    for n in range(depth):
        maps.append(x)
        x = Concatenate()(maps) ##Join a sequence of arrays along an existing axis.
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(features, kernel_size, dilation_rate=dilation_rate, padding=padding)(x)
        dilation_rate *= 2

    
    # probabilities = Conv3D(classes, kernel_size=(1,1,1), activation='softmax')(x)   ### NO NEED IN CLASSIFICATION
    x = Flatten()(x)
    probabilities = Dense(classes, activation='softmax', name='softmax_out')(x)
    print(probabilities)
    # probabilities = x

    return Model(inputs=inputs, outputs=probabilities)