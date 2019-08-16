from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras_config.registries import optimizers


@optimizers.register
class MyAdam(tf.keras.optimizers.Adam):
    pass


if __name__ == '__main__':
    print('See main.py')
