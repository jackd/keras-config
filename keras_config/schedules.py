from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras_config.keras import schedules


@schedules.register
class ClippedExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):

    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate,
                 min_value,
                 staircase=False,
                 name=None):
        super(ClippedExponentialDecay, self).__init__(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )
        self.min_value = min_value

    def __call__(self, step):
        base = super(ClippedExponentialDecay, self).__call__(step)
        return tf.maximum(base, self.min_value)

    def get_config(self):
        config = super(ClippedExponentialDecay, self).get_config()
        config['min_value'] = self.min_value
        return config
