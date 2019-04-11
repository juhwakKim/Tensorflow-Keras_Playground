from __future__ import absolute_import

import tensorflow as tf
from tensorflow import keras


def vgg_block(x):
    conv1 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same' activation='relu' )
    conv1.get_weights()[1]
        

