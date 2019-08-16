from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from keras_config.callbacks import BetterModelCheckpoint
from keras_config.callbacks import LATEST
from keras_config import registry
from keras_config.configurable import Configurable
from keras_config.configurable import reconfigure
from keras_config.keras import optimizers
from keras_config import functions
from keras_config.pipelines import pipelines
from keras_config.problems import problems
from keras_config.yaml import dump_yaml
from keras_config.yaml import load_yaml


class Trainer(Configurable):

    def __init__(self,
                 problem,
                 train_pipeline,
                 validation_pipeline,
                 model_fn,
                 optimizer,
                 chkpt_dir=None,
                 log_dir=None):
        self.problem = problems.get(problem)
        self.train_pipeline = pipelines.get(train_pipeline)
        self.validation_pipeline = pipelines.get(validation_pipeline)
        self.model_fn = functions.get(model_fn)
        self.optimizer = optimizers.get(optimizer)
        if chkpt_dir is not None:
            chkpt_dir = os.path.expanduser(os.path.expandvars(chkpt_dir))
        if log_dir is None:
            log_dir = chkpt_dir
        else:
            log_dir = os.path.expanduser(os.path.expandvars(log_dir))
        self.chkpt_dir = chkpt_dir
        self.log_dir = log_dir
        self._pipelines = {
            'train': self.train_pipeline,
            'validation': self.validation_pipeline
        }

    def get_config(self):
        return dict(problem=problems.serialize(self.problem),
                    train_pipeline=pipelines.serialize(self.train_pipeline),
                    validation_pipeline=pipelines.serialize(
                        self.validation_pipeline),
                    model_fn=functions.serialize(self.model_fn),
                    optimizer=optimizers.serialize(self.optimizer),
                    chkpt_dir=self.chkpt_dir,
                    log_dir=self.log_dir)

    def reconfigure(self, **kwargs):
        for k, v in kwargs:
            if isinstance(v, dict):
                setattr(self, k, reconfigure(getattr(self, k), **v))
            else:
                setattr(self, k, v)

    @classmethod
    def from_log_dir(cls, log_dir, epoch=0):
        path = _config_path(log_dir, epoch)
        if not tf.io.gfile.exists(path):
            raise ValueError('No file at expected config path {}'.format(path))
        return trainers.deserialize(load_yaml(path))

    def clean(self):
        for d in (self.chkpt_dir, self.log_dir):
            if tf.io.gfile.exists(d):
                tf.io.gfile.rmtree(d)

    def _dataset_and_steps(self, split):
        if split == 'test' and split not in self._pipelines:
            split = 'validation'
        pipeline = self._pipelines[split]
        dataset = pipeline(self.problem.get_base_dataset(split))
        num_steps = (self.problem.examples_per_epoch(split) //
                     pipeline.batch_size)
        return dataset, num_steps

    def _get_model_and_callbacks(self):
        problem = self.problem
        model_fn_output = self.model_fn(
            self.train_pipeline.get_output_spec(problem.input_spec),
            problem.output_spec)
        if isinstance(model_fn_output, tf.keras.Model):
            model = model_fn_output
            callbacks = []
        else:
            model, callbacks = model_fn_output
        model.compile(optimizer=self.optimizer,
                      loss=problem.loss,
                      metrics=problem.metrics)
        return model, callbacks

    def train(self, epochs, verbose=True):
        train_ds, train_steps = self._dataset_and_steps('train')
        val_ds, val_steps = self._dataset_and_steps('validation')

        model, callbacks = self._get_model_and_callbacks()
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())

        chkpt_dir = self.chkpt_dir
        if chkpt_dir is None:
            initial_epoch = 0
        else:
            if not tf.io.gfile.isdir(chkpt_dir):
                tf.io.gfile.makedirs(chkpt_dir)
            chkpt_callback = BetterModelCheckpoint(chkpt_dir,
                                                   load_weights_on_restart=True)
            chkpt = chkpt_callback.latest_checkpoint
            if chkpt is None:
                initial_epoch = 0
            else:
                initial_epoch = chkpt_callback.epoch(chkpt)
                chkpt_callback.set_model(model)
                chkpt_callback.restore(chkpt)
            callbacks.append(chkpt_callback)

        log_dir = self.log_dir
        if log_dir is not None:
            if not tf.io.gfile.isdir(log_dir):
                tf.io.gfile.makedirs(log_dir)
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir))
            dump_yaml(self.serialize(), _config_path(log_dir, initial_epoch))

        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_ds,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            initial_epoch=initial_epoch,
        )
        return history

    def evaluate(self, checkpoint=LATEST, verbose=True):
        val_ds, val_steps = self._dataset_and_steps('validation')
        model, callbacks = self._get_model_and_callbacks()
        chkpt_callback = BetterModelCheckpoint(self.chkpt_dir)
        chkpt_callback.set_model(model)
        chkpt_callback.restore(checkpoint)
        model.evaluate(val_ds,
                       steps=val_steps,
                       callbacks=callbacks,
                       verbose=verbose)


trainers = registry.Registry('trainers', registry.subclass_validator(Trainer))
trainers.register(Trainer)


def _config_path(log_dir, epoch=0):
    return os.path.join(log_dir, 'config-{}.yaml'.format(epoch))


get = trainers.get
deserialize = trainers.deserialize
serialize = trainers.serialize
register = trainers.register
