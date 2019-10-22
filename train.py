#!/usr/bin/env python
# -*- coding:utf8 -*-

import sys
import os
from transformer import *
import tensorflow as tf
from tensorflow.python.client import device_lib
from random import shuffle
import common_utils
from importlib import import_module

tf.logging.set_verbosity(tf.logging.INFO)

## @bao dynamic import 
assert len(sys.argv) == 2, 'python xx.py config_file'
_, config_file = sys.argv
if config_file.endswith('.py'):
    config_file = config_file.rsplit('.', 1)[0]
tf.logging.info('Using config from %s'%config_file)
in_config = import_module(config_file)
params = getattr(in_config, 'params')


def output_model(model, params, output_path=None):
    import tensorflow_hub as tf_hub
    if output_path == None:
        output_path = "output_model_" + config_file

    ## transformer_export_fn is imported from transformer.py
    module_spec = tf_hub.create_module_spec(transformer_export_fn, 
                                        [(set(), {'params':params})])
    with tf.Graph().as_default():
        module = tf_hub.Module(module_spec)
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                                    tf.train.latest_checkpoint(params["save_path"]), 
                                    module.variable_map)
        with tf.Session() as session:
          init_fn(session)
          module.export(output_path, session=session)

class EvaluationListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, eval_input_fn, max_step, min_step, model_params):
        self._estimator = estimator  # # type: tf.estimator.Estimator
        self._eval_input_fn = eval_input_fn  # return data iterator
        self._max_step = max_step
        self._min_step = min_step
        self._model_params = model_params

    def _should_trigger(self, global_steps):
        if global_steps > self._min_step:
            return True
        else:
            return False

    def before_save(self, session, global_step_value):
        pass

    def after_save(self, session, global_step_value):
        if self._should_trigger(global_step_value):
            tf.logging.info("Importing parameters for evaluation at step {0}...".format(global_step_value))
            for eval_name in self._eval_input_fn:
                tf.logging.info("============== Evaluating %s ============="%eval_name)
                eval_result = self._estimator.evaluate(self._eval_input_fn[eval_name],
                                         checkpoint_path=tf.train.latest_checkpoint(self._output_dir))
                tf.logging.info("Done.")
                tf.logging.info(eval_result)

        if global_step_value >= self._max_step:
            ## export model
            if params['init_emb'] != None:
                params['init_emb'] = None ## reset or will cause problem sometimes
            output_model(self._estimator, self._model_params)
            exit()

def main(_):
    gpu_devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if len(gpu_devices) == 0:
        params['num_gpus'] = 1
    else:
        params['num_gpus'] = len(gpu_devices)
    print(params)
    transformer = tf.estimator.Estimator(model_fn=transformer_model_fn,
                                        config=tf.estimator.RunConfig(
                                                save_checkpoints_steps=params['save_checkpoints_steps']//params['num_gpus'], 
                                                log_step_count_steps=10000//params['num_gpus'],
                                                keep_checkpoint_max=params['keep_checkpoint_max']),
                                        model_dir=params["save_path"],
                                        params=params)

    def train_input_fn():
        return input_fn(
            params['train_src'],
            params['vocab_src'],
            src_vocab_size=params["src_vocab_size"],
            batch_size_words=params['train_batch_size_words'],
            max_len=params['max_len'],
            num_gpus=params['num_gpus'],
            is_train=True
        )

    eval_funcs = {}
    ## @bao for-loop has some bugs
    #for dset in params["dev_src"]:
    #    print('Add listener eval func for %s'%dset)
    #    eval_funcs[dset] = lambda : input_fn({dset: params['dev_src'][dset]},
    #                                         {dset: params['vocab_src'][dset]},
    #                                         src_vocab_size={dset: params['src_vocab_size'][dset]},
    #                                         batch_size=params["infer_batch_size"],
    #                                         is_train=False)

    ## @bao hardcode eval function
    if 'en' in params['dev_src']:
        eval_funcs['en'] = lambda : input_fn({'en': params['dev_src']['en']},
                                             {'en': params['vocab_src']['en']},
                                             src_vocab_size={'en': params['src_vocab_size']['en']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)
    if 'es' in params['dev_src']:
        eval_funcs['es'] = lambda : input_fn({'es': params['dev_src']['es']},
                                             {'es': params['vocab_src']['es']},
                                             src_vocab_size={'es': params['src_vocab_size']['es']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)
    if 'nl' in params['dev_src']:
        eval_funcs['nl'] = lambda : input_fn({'nl': params['dev_src']['nl']},
                                             {'nl': params['vocab_src']['nl']},
                                             src_vocab_size={'nl': params['src_vocab_size']['nl']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)
    if 'zh' in params['dev_src']:
        eval_funcs['zh'] = lambda : input_fn({'zh': params['dev_src']['zh']},
                                             {'zh': params['vocab_src']['zh']},
                                             src_vocab_size={'zh': params['src_vocab_size']['zh']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)
    if 'de' in params['dev_src']:
        eval_funcs['de'] = lambda : input_fn({'de': params['dev_src']['de']},
                                             {'de': params['vocab_src']['de']},
                                             src_vocab_size={'de': params['src_vocab_size']['de']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)

    if 'da' in params['dev_src']:
        eval_funcs['da'] = lambda : input_fn({'da': params['dev_src']['da']},
                                             {'da': params['vocab_src']['da']},
                                             src_vocab_size={'da': params['src_vocab_size']['da']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)

    if 'el' in params['dev_src']:
        eval_funcs['el'] = lambda : input_fn({'el': params['dev_src']['el']},
                                             {'el': params['vocab_src']['el']},
                                             src_vocab_size={'el': params['src_vocab_size']['el']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)

    if 'pt' in params['dev_src']:
        eval_funcs['pt'] = lambda : input_fn({'pt': params['dev_src']['pt']},
                                             {'pt': params['vocab_src']['pt']},
                                             src_vocab_size={'pt': params['src_vocab_size']['pt']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)

    if 'it' in params['dev_src']:
        eval_funcs['it'] = lambda : input_fn({'it': params['dev_src']['it']},
                                             {'it': params['vocab_src']['it']},
                                             src_vocab_size={'it': params['src_vocab_size']['it']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)

    if 'sv' in params['dev_src']:
        eval_funcs['sv'] = lambda : input_fn({'sv': params['dev_src']['sv']},
                                             {'sv': params['vocab_src']['sv']},
                                             src_vocab_size={'sv': params['src_vocab_size']['sv']},
                                             batch_size=params["infer_batch_size"],
                                             is_train=False)

    eval_listener = EvaluationListener(
                                        estimator=transformer,
                                        eval_input_fn=eval_funcs,
                                        max_step=params['max_step'],
                                        min_step=params["min_start_step"],
                                        model_params=params
                                    )

    transformer.train(train_input_fn, saving_listeners=[eval_listener])


if __name__ == '__main__':
    tf.app.run()

