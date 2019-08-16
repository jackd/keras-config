from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_config.configurable import Configurable
from keras_config.configurable import Configurer
from keras_config import registry
from keras_config import keras as _keras
import six


class Objective(Configurable):

    def __init__(self, name, mode='max'):
        self.name = name
        self.mode = mode

    def get_config(self):
        return dict(name=self.name, mode=self.mode)

    @classmethod
    def get(self, identifier):
        if isinstance(identifier, Objective):
            return identifier
        if isinstance(identifier, (list, tuple)):
            return Objective(*identifier)
        elif isinstance(identifier, dict):
            return Objective(**identifier)
        elif isinstance(identifier, six.string_types):
            return Objective(identifier)
        else:
            raise TypeError(
                'Cannot convert identifier {} into an Objective'.format(
                    identifier))


class Problem(Configurable):

    def __init__(self,
                 loss,
                 metrics=(),
                 objective=None,
                 input_spec=None,
                 output_spec=None):
        self.loss = _keras.losses.get(loss)
        self.metrics = [_keras.metrics.get(m) for m in metrics]
        if objective is None and len(self.metrics) > 0:
            objective = self.metrics[0].name
        self.objective = Objective.get(objective)
        self.input_spec = get_input_spec(input_spec)
        self.output_spec = get_input_spec(output_spec)

    @abc.abstractmethod
    def _examples_per_epoch(self, split):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_base_dataset(self, split):
        raise NotImplementedError

    def examples_per_epoch(self, split=tfds.Split.TRAIN, batch_size=None):
        return tf.nest.map_structure(self._examples_per_epoch, split)

    def get_base_dataset(self, split=tfds.Split.TRAIN):
        return tf.nest.map_structure(self._get_base_dataset, split)

    def get_config(self):
        objective = self.objective
        return dict(
            loss=_keras.losses.serialize(self.loss),
            metrics=[_keras.metrics.serialize(m) for m in self.metrics],
            objective=None if objective is None else objective.get_config(),
            input_spec=get_input_spec_config(self.input_spec),
            output_spec=get_input_spec_config(self.output_spec))


problems = registry.Registry('problems', registry.subclass_validator(Problem))


def get_input_spec_config(input_spec):
    if input_spec is None:
        return None
    return dict(dtype=repr(input_spec.dtype)[3:],
                shape=input_spec.shape,
                ndim=input_spec.ndim,
                max_ndim=input_spec.max_ndim,
                min_ndim=input_spec.min_ndim,
                axes=input_spec.axes)


def get_input_spec(identifier):
    if identifier is None or isinstance(identifier, tf.keras.layers.InputSpec):
        return identifier
    elif isinstance(identifier, dict):
        return tf.keras.layers.InputSpec(**identifier)
    else:
        raise TypeError(
            'Cannot convert value {} to InputSpec'.format(identifier))


# input_spec_configurer = Configurer(
#     from_config=lambda config: tf.keras.layers.InputSpec(**config),
#     get_config=get_input_spec_config)

# registry.misc.register(tf.keras.layers.InputSpec, input_spec_configurer)


@problems.register
class TfdsProblem(Problem):

    def __init__(self,
                 builder,
                 loss,
                 metrics=(),
                 objective=None,
                 input_spec=None,
                 output_spec=None,
                 as_supervised=True,
                 split_map=None):
        if isinstance(builder, six.string_types):
            builder = tfds.builder(builder)
        self.builder = builder

        self.as_supervised = as_supervised
        if input_spec is None or output_spec is None:
            info = self.builder.info
            inp, out = (info.features[k] for k in info.supervised_keys)
            if input_spec is None:
                input_spec = tf.keras.layers.InputSpec(shape=inp.shape,
                                                       dtype=inp.dtype)
            if output_spec is None:
                if hasattr(out, 'num_classes'):
                    # classification
                    shape = out.shape + (out.num_classes,)
                    dtype = tf.float32
                else:
                    shape = out.shape
                    dtype = out.dtype
                output_spec = tf.keras.layers.InputSpec(shape=shape,
                                                        dtype=dtype)
        if split_map is None:
            split_map = {}
        self.split_map = split_map
        super(TfdsProblem, self).__init__(
            loss=loss,
            metrics=metrics,
            objective=objective,
            input_spec=input_spec,
            output_spec=output_spec,
        )

    def get_config(self):
        config = super(TfdsProblem, self).get_config()
        builder = self.builder
        config['builder'] = (builder.name if
                             builder.builder_config is None else '{}/{}'.format(
                                 builder.name, builder.builder_config.name))
        config['split_map'] = self.split_map
        config['as_supervised'] = self.as_supervised
        return config

    def _split(self, split):
        return self.split_map.get(split, split)

    def _examples_per_epoch(self, split):
        split = self._split(split)

        def get(split):
            return self.builder.info.splits[split].num_examples

        if isinstance(split, (tfds.core.splits.NamedSplit,) + six.string_types):
            return get(split)
        else:
            # fractional split?
            # https://github.com/tensorflow/datasets/tree/master/docs/splits.md
            acc = 0
            for k, (start, end) in split.items():
                percent = round((end - start) * 100) / 100
                acc += round(get(k) * percent)
            return acc

    def _get_base_dataset(self, split):
        split = self._split(split)
        if isinstance(split, dict):
            RI = tfds.core.tfrecords_reader.ReadInstruction
            ri = None
            for k, (from_, to) in split.items():
                nex = RI(k, from_=from_ * 100, to=to * 100, unit='%')
                if ri is None:
                    ri = nex
                else:
                    ri = ri + nex
            split = ri

        return self.builder.as_dataset(split=split,
                                       as_supervised=self.as_supervised)


get = problems.get
deserialize = problems.deserialize
serialize = problems.serialize
register = problems.register
