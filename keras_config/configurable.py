from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import importlib
import six


class Configurable(abc.ABC):

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def serialize(self):
        return dict(class_name=self.__class__.__name__,
                    module=self.__module__,
                    config=self.get_config())


def reconfigure(configurable, **updates):
    if hasattr(configurable, 'reconfigure'):
        return configurable.reconfigure(**updates)
    elif isinstance(configurable, dict):
        for k, v in updates.items():
            if k in configurable:
                if isinstance(v, dict):
                    configurable[k] = reconfigure(configurable[k], **v)
                else:
                    configurable[k] = v
            else:
                configurable[k] = v
        return configurable
    else:
        config = configurable.get_config()
        config.update(updates)
        return configurable.__class__.from_config(config)


class Configurer(object):

    def __init__(self, from_config, get_config):
        self.from_config = from_config
        self.get_config = get_config
