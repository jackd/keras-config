from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import importlib
import six
import types
from keras_config import registry
from keras_config.configurable import Configurable

ATOMIC_TYPES = (int, float) + six.string_types

# class FunctionRegistry(registry.Registry):

#     def deserialize(self, identifier):
#         if isinstance(identifier, types.FunctionType):
#             return ConfigurableFunction(identifier)
#         elif isinstance(identifier, functools.partial):
#             return ConfigurablePartial(identifier)
#         elif isinstance(identifier, six.string_types):
#             raise ValueError('Cannot get function with just a string')
#         else:
#             return super(FunctionRegistry, self).deserialize(identifier)

#     def serialize(self, instance):
#         if isinstance(instance, types.FunctionType):
#             instance = ConfigurableFunction(instance)
#         elif isinstance(instance, functools.partial):
#             instance = ConfigurablePartial(instance)
#         return super(FunctionRegistry, self).serialize(instance)

# functions = FunctionRegistry('functions')
functions = registry.Registry('functions')


@functions.register
class ConfigurableFunction(Configurable):

    def __init__(self, func):
        for attr in ('__name__', '__module__'):
            if not hasattr(func, attr):
                raise ValueError(
                    'func must have attr {} to be configurable'.format(attr))
        if func.__name__ == '<lambda>':
            raise ValueError('Cannot wrap lambda functions as configurables')
        self._func = func

    def get_config(self):
        return dict(
            name=self._func.__name__,
            module=self._func.__module__,
        )

    @property
    def func(self):
        return self._func

    @classmethod
    def from_config(self, config):
        func = getattr(importlib.import_module(config['module']),
                       config['name'])
        return ConfigurableFunction(func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def _update_args_and_kwargs(args0, kwargs0, args1, kwargs1):
    # mutates kwargs1
    args = args0 + args1
    for k in kwargs0:
        if k in kwargs1:
            raise ValueError('Duplicate value for keyword "{}"'.format(k))
    kwargs1.update(kwargs0)
    return args, kwargs1


def _is_registerable_fn(fn):
    return (isinstance(fn, types.FunctionType) or
            isinstance(fn, functools.partial) and _is_registerable_fn(
                fn.func) and all(_is_registerable_arg(v) for v in fn.args) and
            all(_is_registerable_arg(v) for v in fn.keywords.values()))


def _is_registerable_arg(x):
    return (isinstance(x, ATOMIC_TYPES) or isinstance(x, (list, tuple)) and
            all(_is_registerable_arg(xi) for xi in x) or
            isinstance(x, dict) and all(
                _is_registerable_arg(v) and isinstance(k, six.string_types)
                for k, v in x.items()) or _is_registerable_fn(x))


@functions.register
class ConfigurablePartial(Configurable):

    def __init__(self, func, *args, **kwargs):
        if isinstance(func, dict):
            func = ConfigurableFunction.from_config(func).func
        if not callable(func):
            raise ValueError(
                'ConfigurablePartial requires callable as first argument')

        while isinstance(func, functools.partial):
            args, kwargs = _update_args_and_kwargs(func.args, func.keywords,
                                                   args, kwargs)
            func = func.func

        for arg in args:
            if not _is_registerable_arg(arg):
                raise ValueError('Unconfigurable argument {}'.format(arg))
        for key, arg in kwargs.items():
            if not _is_registerable_arg(arg):
                raise ValueError('Unconfigurable argument {} for key {}'.format(
                    arg, key))

        self._partial = functools.partial(func, *args, **kwargs)

    @property
    def args(self):
        return self._partial.args

    @property
    def keywords(self):
        return self._partial.keywords.copy()

    @property
    def func(self):
        return self._partial.func

    def __call__(self, *args, **kwargs):
        return self._partial(*args, **kwargs)

    def get_config(self):
        return dict(
            func=ConfigurableFunction(self._partial.func).get_config(),
            args=self._partial.args,
            keywords=self._partial.keywords.copy(),
        )

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        return ConfigurablePartial(config.pop('func'), *config['args'],
                                   **config['keywords'])


functions.register(types.FunctionType, ConfigurableFunction)
functions.register(functools.partial, ConfigurablePartial)

get = functions.get
deserialize = functions.deserialize
serialize = functions.serialize
register = functions.register
