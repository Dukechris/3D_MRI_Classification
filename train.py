#!/usr/bin/env python

from __future__ import division, print_function

import os
import argparse # 
import logging  #

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import np_utils

from DataLoader import Dataset
from Model import Models



def select_optimizer(optimizer_name, optimizer_args):
    return 0
    
#/Users/zhangxiao/Desktop/3D_MRI_Classification/Data
#/home/xzhang/kerasLab/3D_MRI_Classification/Data
def train():
    train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch = Dataset.create_generators(
            '/Users/zhangxiao/Desktop/3D_MRI_Classification/Data', 4,
            validation_split=0.2,
            shuffle_train_val=True,
            shuffle=True,
            seed=0)

    images, labels = next(train_generator)
    _, height, width, length, channels = images.shape
    print(images.shape, labels)
    classes_num = 2 #len(set(labels.flatten()))

    model = Models.dilated_densenet(height=height, width=width, length=length, channels=channels, 
        classes=classes_num, features=32, depth=3, padding='same',
        temperature=1.0, batchnorm=False, dropout=0.0)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])


    callbacks = []
    # x_train, y_train = Dataset.load_images('/home/xzhang/kerasLab/3D_MRI_Classification/Data')
    # y_train = np_utils.to_categorical(y_train, 2)
    model.fit_generator(train_generator,
        epochs=20,
        steps_per_epoch=train_steps_per_epoch,
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks, verbose=2)

    # y_train = np_utils.to_categorical(y_train, 2)
    # print(y_train)
    # model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=2)


    return 0
    

if __name__ == '__main__':
    train()
