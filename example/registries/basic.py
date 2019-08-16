from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras_config.registries import optimizers


@optimizers.register
class MyAdam(tf.keras.optimizers.Adam):
    pass


print(optimizers.get('Adam'))
print(optimizers.get('MyAdam'))
serialized = optimizers.serialize(optimizers.get(MyAdam()))
print(serialized)
print(optimizers.deserialize(serialized))
