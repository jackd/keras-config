from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from keras_config.trainers import Trainer
from keras_config.yaml import load_yaml
from keras_config.session_options import SessionOptions

SessionOptions().configure_session()

trainer = Trainer.from_log_dir(
    os.path.join(os.path.dirname(__file__), './model'))

trainer.train(epochs=15, verbose=True)
