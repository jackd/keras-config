from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from keras_config.registries import optimizers

with open('my_opt.yaml', 'r') as fp:
    spec = yaml.load(fp, yaml.SafeLoader)

print(optimizers.get(spec))
