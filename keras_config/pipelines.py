from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_config.configurable import Configurable
from keras_config import registry
from keras_config.functions import functions
from keras_config.problems import get_input_spec
from keras_config.problems import get_input_spec_config

AUTOTUNE = tf.data.experimental.AUTOTUNE
NO_REPEAT = 'NO_REPEAT'


class Pipeline(Configurable):

    def __init__(self,
                 batch_size=None,
                 repeats=NO_REPEAT,
                 shuffle_buffer=None,
                 map_fn=None,
                 prefetch_buffer=AUTOTUNE,
                 num_parallel_calls=AUTOTUNE,
                 output_spec=None,
                 output_spec_fn=None):
        if output_spec is not None and output_spec_fn is not None:
            raise ValueError(
                'Only one of output_spec or output_spec_fn should be specified')
        self.batch_size = batch_size
        self.repeats = repeats
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.num_parallel_calls = num_parallel_calls
        self.map_fn = functions.get(map_fn)
        if isinstance(output_spec, dict):
            if 'class_name' in output_spec:
                output_spec = functions.get(output_spec)
            else:
                output_spec = get_input_spec(output_spec)

        self.output_spec = get_input_spec(output_spec)
        self.output_spec_fn = functions.get(output_spec_fn)

    def get_output_spec(self, input_spec):
        if self.output_spec is not None:
            return self.output_spec
        elif self.output_spec_fn is not None:
            return self.output_spec_fn(input_spec)
        else:
            return input_spec

    def preprocess_dataset(self, dataset):
        if self.repeats != NO_REPEAT:
            dataset = dataset.repeat()
        if self.shuffle_buffer is not None:
            dataset = dataset.shuffle(self.shuffle_buffer)
        if self.map_fn is not None:
            dataset = dataset.map(self.map_fn, self.num_parallel_calls)
        if self.batch_size is not None:
            dataset = dataset.batch(self.batch_size)
        if self.prefetch_buffer:
            dataset = dataset.prefetch(self.prefetch_buffer)
        return dataset

    def __call__(self, dataset):
        return self.preprocess_dataset(dataset)

    def get_generator(self, dataset_fn):
        graph = tf.Graph()
        with graph.as_default():  # pylint: disable=not-context-manager
            dataset = self.preprocess_dataset(dataset_fn())
        return tfds.as_numpy(dataset, graph=graph)

    def get_config(self):
        return dict(batch_size=self.batch_size,
                    repeats=self.repeats,
                    shuffle_buffer=self.shuffle_buffer,
                    map_fn=functions.serialize(self.map_fn),
                    prefetch_buffer=self.prefetch_buffer,
                    num_parallel_calls=self.num_parallel_calls,
                    output_spec=get_input_spec_config(self.output_spec),
                    output_spec_fn=functions.serialize(self.output_spec_fn))


pipelines = registry.Registry('pipelines',
                              registry.subclass_validator(Pipeline))

pipelines.register(Pipeline)

get = pipelines.get
deserialize = pipelines.deserialize
serialize = pipelines.serialize
register = pipelines.register
