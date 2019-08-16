from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras_config.yaml import load_yaml

z = 'defined above'

path = os.path.join(os.path.dirname(__file__), 'spec.yaml')
data = load_yaml(path, module_objects=globals())
print(data)
