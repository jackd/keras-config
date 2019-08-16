"""
Registries for keras structures:
# * callbacks
* constraints
* optimizers
* layers
* losses
* metrics
* regularizers
* schedules
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_config.registry import Registry
from keras_config.registry import subclass_validator
import tensorflow as tf
keras = tf.keras


def _update_registry(registry, module):
    for k in dir(module):
        value = getattr(module, k)
        if registry.is_registerable(value):
            registry.register(value)


def _register_module(module, name, validator):
    registry = Registry(name, validator)
    _update_registry(registry, module)
    return registry


class OptimizerRegistry(Registry):

    def deserialize(self, identifier):
        with keras.utils.custom_object_scope(schedules):
            return super(OptimizerRegistry, self).deserialize(identifier)


optimizers = OptimizerRegistry('optimizers',
                               subclass_validator(keras.optimizers.Optimizer))
_update_registry(optimizers, keras.optimizers)
losses = _register_module(keras.losses, 'losses',
                          subclass_validator(keras.losses.Loss))
metrics = _register_module(keras.metrics, 'metrics',
                           subclass_validator(keras.metrics.Metric))
regularizers = _register_module(
    keras.regularizers, 'regularizers',
    subclass_validator(keras.regularizers.Regularizer))

constraints = _register_module(keras.constraints, 'constraints',
                               subclass_validator(keras.constraints.Constraint))

schedules = _register_module(
    keras.optimizers.schedules, 'schedules',
    subclass_validator(keras.optimizers.schedules.LearningRateSchedule))

layers = _register_module(keras.layers, 'layers',
                          subclass_validator(keras.layers.Layer))

# callbacks = _register_module(keras.callbacks, 'callbacks',
#                              subclass_validator(keras.callbacks.Callback))

__all__ = [
    # 'callbacks',
    'layers',
    'losses',
    'metrics',
    'optimizers',
    'regularizers',
    'schedules',
    'constraints',
]
