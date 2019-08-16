from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import importlib
import six


class Registry(collections.Mapping):

    def __init__(self, name, validator=lambda x: True):
        self._registry = {}
        self._validator = validator
        self._name = name

    def __getitem__(self, key):
        return self._registry[key]

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def __contains__(self, key):
        return key in self._registry

    def _assert_registerable(self, cls, configurer):
        if configurer is None:
            configurer = cls
        if not isinstance(cls, type):
            raise ValueError(
                'Registered values must be types/classes, got {}'.format(cls))
        if not hasattr(configurer, 'from_config'):
            raise ValueError(
                'Registered values must have `from_config` members, '
                'but {} does not'.format(configurer))
        if not hasattr(configurer, 'get_config'):
            raise ValueError(
                'Registered values must have `from_config` members, '
                'but {} does not'.format(configurer))
        if not self._validator(configurer):
            raise ValueError(
                'Registration of class {} failed validation for registry {}'.
                format(configurer, self._name))

    def is_registerable(self, cls, configurer=None):
        try:
            self._assert_registerable(cls, configurer)
            return True
        except ValueError:
            return False

    def register(self, cls, configurer=None):
        if configurer is None:
            configurer = cls
        self._assert_registerable(cls, configurer)
        self._registry[cls.__name__] = configurer
        return cls

    def get(self, identifier):
        if identifier.__class__.__name__ in self._registry:
            return identifier
        else:
            return self.deserialize(identifier)

    def deserialize(self, identifier):
        if identifier is None:
            return None
        elif isinstance(identifier, six.string_types):
            return self[identifier]()
        elif identifier.__class__.__name__ in self:
            return identifier
        elif isinstance(identifier, dict):
            if 'class_name' not in identifier:
                raise ValueError(
                    'Missing required key "class_name" from registry {}'.format(
                        self._name))
            class_name = identifier['class_name']
            if class_name not in self:
                if 'module' not in identifier:
                    raise ValueError(
                        'Unrecognized class_name {} and no "module" key found '
                        'for deserializing {}'.format(class_name, self._name))
                importlib.import_module(identifier['module'])
                if class_name not in self:
                    raise ValueError(
                        'Unrecognized class_name {} for deserializing {}'.
                        format(class_name, self._name))

            # class_name must be present
            return self[class_name].from_config(identifier.get('config', {}))
        else:
            raise ValueError('Cannot deserialize {} using registry {}'.format(
                identifier, self._name))

    def serialize(self, instance):
        if instance is None:
            return None
        if instance.__class__.__name__ in self:
            if not hasattr(instance, 'get_config'):
                instance = self[instance.__class__.__name__](instance)
            return dict(class_name=instance.__class__.__name__,
                        config=instance.get_config(),
                        module=instance.__class__.__module__)
        else:
            raise ValueError('instance {} not in registry {}'.format(
                instance, self._name))

    def _export_globals(self,
                        globals_dict,
                        keys=('get', 'serialize', 'deserialize', 'register')):
        for key in keys:
            if key in globals_dict:
                raise KeyError(
                    'Cannot export global "{}" - already exists'.format(key))
            globals_dict[key] = getattr(self, key)


def subclass_validator(cls):

    def validate(proposed):
        if not issubclass(proposed, cls):
            raise TypeError(
                'Can only register classes subclassing {}, but {} does not'.
                format(cls.__name__, proposed))
        return True

    return validate


def has_attrs_validator(*attrs):

    def validate(proposed):
        for attr in attrs:
            if not hasattr(proposed, attr):
                raise ValueError(
                    'class {} missing required attribute {}'.format(
                        proposed.__name__, attr))

    return validate


# misc = Registry('misc')
