from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import functools
from keras_config.problems import TfdsProblem
from keras_config.problems import get_input_spec_config
from keras_config.pipelines import Pipeline
from keras_config import registries as reg
from keras_config.layers import VariableMomentumBatchNormalization
from keras_config.callbacks import BatchNormMomentumScheduler
from keras_config.schedules import ClippedExponentialDecay
from keras_config import trainers


def exponential_decay(step,
                      initial_value,
                      decay_steps,
                      decay_rate,
                      min_value=None,
                      staircase=False):
    exponent = step / decay_steps
    if staircase:
        exponent = np.floor(exponent)
    value = initial_value * decay_rate**exponent
    if min_value is not None:
        value = max(value, min_value)
    return value


def complementary_exponential_decay(step,
                                    initial_value,
                                    decay_steps,
                                    decay_rate,
                                    max_value=0.99,
                                    staircase=False):
    return 1 - exponential_decay(step,
                                 1 - initial_value,
                                 decay_steps,
                                 decay_rate,
                                 None if max_value is None else 1 - max_value,
                                 staircase=staircase)


def complementary_exponential_decay_fn(initial_value,
                                       decay_steps,
                                       decay_rate,
                                       max_value=0.99,
                                       staircase=False):
    return functools.partial(
        complementary_exponential_decay,
        initial_value=initial_value,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        max_value=max_value,
        staircase=staircase,
    )


def get_cifar_model(input_spec,
                    output_spec,
                    training=None,
                    conv_filters=(16, 32),
                    dense_units=(),
                    activation='relu'):
    num_classes = output_spec.shape[-1]
    inputs = tf.keras.layers.Input(shape=input_spec.shape,
                                   dtype=input_spec.dtype)
    x = inputs
    for f in conv_filters:
        x = tf.keras.layers.Conv2D(f, 3)(x)
        x = VariableMomentumBatchNormalization()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = tf.keras.layers.Dense(u)(x)
        x = VariableMomentumBatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    logits = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    schedule = complementary_exponential_decay_fn(0.5, 10, 0.5)
    callbacks = [BatchNormMomentumScheduler(schedule)]
    return model, callbacks


problem = TfdsProblem(
    'cifar100',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
    ],
    split_map={'validation': 'test'})
problem.builder.download_and_prepare()


def val_map_fn(image, labels):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, labels


def train_map_fn(image, labels, saturation_delta=0.1, hue_max_delta=0.1):
    image = tf.cast(image, tf.float32)
    image = tf.image.random_saturation(image, 1 - saturation_delta,
                                       1 + saturation_delta)
    image = tf.image.random_hue(image, max_delta=hue_max_delta)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image, labels


def as_float(input_spec):
    config = get_input_spec_config(input_spec)
    config['dtype'] = tf.float32
    return tf.keras.layers.InputSpec(**config)


def get_trainer():
    chkpt_dir = os.path.join(os.path.dirname(__file__), 'model')
    BATCH_SIZE = 32
    train_pipeline = Pipeline(BATCH_SIZE,
                              repeats=None,
                              shuffle_buffer=1024,
                              map_fn=functools.partial(train_map_fn,
                                                       saturation_delta=0.1,
                                                       hue_max_delta=0.1),
                              output_spec_fn=as_float)
    validation_pipeline = Pipeline(BATCH_SIZE,
                                   repeats=None,
                                   map_fn=val_map_fn,
                                   output_spec_fn=as_float)

    model_fn = functools.partial(get_cifar_model, dense_units=(128,))

    optimizer = tf.keras.optimizers.Adam(learning_rate=ClippedExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.5,
        min_value=1e-5,
    ))

    trainer = trainers.Trainer(problem,
                               train_pipeline,
                               validation_pipeline,
                               model_fn,
                               optimizer,
                               chkpt_dir=chkpt_dir)
    return trainer
