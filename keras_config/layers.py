from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras_config.keras import layers


@layers.register
class VariableMomentumBatchNormalization(tf.keras.layers.BatchNormalization):

    def build(self, input_shape):
        if not self.built:
            self.initial_momentum = self.momentum
            self.momentum = self.add_weight(
                'momentum',
                initializer=tf.keras.initializers.constant(self.momentum),
                # initializer=tf.keras.initializers.constant(
                #     self.momentum * np.ones(shape=input_shape[-1])),
                trainable=False)
            super(VariableMomentumBatchNormalization, self).build(input_shape)

    def get_config(self):
        config = super(VariableMomentumBatchNormalization, self).get_config()
        if self.built:
            # actual momentum is a variable and should be saved in weights
            config['momentum'] = self.initial_momentum
        return config


get = layers.get
deserialize = layers.deserialize
serialize = layers.serialize
register = layers.register
