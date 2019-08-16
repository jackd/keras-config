from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from keras_config.yaml import Loader
import os

with open(os.path.join(os.path.dirname(__file__), 'base.yaml'), 'r') as fp:
    out = yaml.load(fp, Loader)

print(out)
