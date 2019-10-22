import logging

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.python.summary.writer import writer_cache


def default_options(params, mode=None):
    if "log_dir" not in params:
        params["log_dir"] = None

    if mode == "train":
        if "l2_regularization" not in params:
            params["l2_regularization"] = 0.0

    if "num_cpus" not in params:
        params["num_cpus"] = 4

    if "min_eval_step" not in params:
        params["min_eval_step"] = 5

    return params


def exist_or_new(dir_name):
    """If directory exists, return True, otherwise make a new directory."""
    if tf.gfile.Exists(dir_name):
        return True
    else:
        tf.gfile.MkDir(dirname=dir_name)


def close_dropout(params):
    """If there is any option on dropout, set it to 0.0"""
    params_wo_drop = dict()

    for k, v in params.items():
        if 'dropout' in k:
            params_wo_drop[k] = 0.0
        else:
            params_wo_drop[k] = v

    return params_wo_drop


def add_log_filehandler(log_path=None):
    """Redirect tensorflow log to file elsewhere"""
    if log_path is None:
        return

    if tf.gfile.Exists(log_path):
        tf.gfile.Remove(log_path)

    log = logging.getLogger("tensorflow")

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def write_dict_to_summary(output_dir, dictionary, current_global_step):
    """Writes a `dict` into summary file in given output directory.

    Args:
      output_dir: `str`, directory to write the summary file in.
      dictionary: the `dict` to be written to summary file.
      current_global_step: `int`, the current global step.
    """

    summary_writer = writer_cache.FileWriterCache.get(output_dir)
    summary_proto = summary_pb2.Summary()
    for key in dictionary:
        if dictionary[key] is None:
            continue
        if key == 'global_step':
            continue
        if (isinstance(dictionary[key], np.float32) or
                isinstance(dictionary[key], float)):
            summary_proto.value.add(tag=key, simple_value=float(dictionary[key]))
        elif (isinstance(dictionary[key], np.int64) or
              isinstance(dictionary[key], np.int32) or
              isinstance(dictionary[key], int)):
            summary_proto.value.add(tag=key, simple_value=int(dictionary[key]))
        else:
            tf.logging.warn(
                'Skipping summary for %s, must be a float, np.float32, np.int64, '
                'np.int32 or int or a serialized string of Summary.', key)

    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()
