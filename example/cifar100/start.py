from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_config.session_options import SessionOptions
from cifar_trainer import get_trainer

SessionOptions().configure_session()

trainer = get_trainer()
trainer.clean()
trainer.train(epochs=5, verbose=True)
