from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
import os.path
import six
import yaml

_dtypes = {
    'float32': tf.float32,
    'float64': tf.float64,
    'int8': tf.int8,
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64,
    'uint8': tf.uint8,
    'uint16': tf.uint16,
    'uint32': tf.uint32,
    'uint64': tf.uint64,
}


class Loader(yaml.SafeLoader):
    """
    https://interviewbubble.com/solved-how-can-i-import-an-yaml-file-inside-another-yaml-file/
    """

    def __init__(self, stream, module_objects={}):
        self._root = os.path.split(stream.name)[0]
        self._module_objects = module_objects
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

    def eval(self, node):
        return eval(self.construct_yaml_str(node), self._module_objects)

    # def dtype(self, node):
    #     return _dtypes[self.construct_yaml_str(node)]


def cached_loader_factory(cache):

    def get_loader(stream):
        return Loader(stream, cache)

    return get_loader


Loader.add_constructor('!include', Loader.include)
Loader.add_constructor('!eval', Loader.eval)
# Loader.add_constructor('!dtype', Loader.dtype)


class Dumper(yaml.SafeDumper):

    def represent_tuple(self, data):
        return self.represent_list(list(data))

    def represent_dtype(self, data):
        # return self.represent_scalar('!dtype', format(repr(data)[3:]))
        return self.represent_str(repr(data)[3:])


Dumper.add_representer(tuple, Dumper.represent_tuple)
Dumper.add_representer(tf.dtypes.DType, Dumper.represent_dtype)


def load_yaml(
        path_or_stream,
        module_objects=None,
        Loader=Loader,
):
    if module_objects is not None:
        loader = functools.partial(Loader, module_objects=module_objects)
    else:
        loader = Loader
    if isinstance(path_or_stream, six.string_types):
        with open(path_or_stream, 'r') as fp:
            return yaml.load(fp, loader)
    else:
        return yaml.load(path_or_stream, loader)


def dump_yaml(data, path_or_stream, Dumper=Dumper):
    if isinstance(path_or_stream, six.string_types):
        with open(path_or_stream, 'w') as fp:
            yaml.dump(data, fp, Dumper)
    else:
        yaml.dump(data, path_or_stream, Dumper)
