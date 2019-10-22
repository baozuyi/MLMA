#!/usr/bin/env python
# -*- coding:utf8 -*-

import tensorflow as tf

def input_fn(src_file,
             src_vocab_file,
             src_vocab_size=None,
             num_buckets=20,
             max_len=100,
             batch_size=200,
             batch_size_words=4096,
             num_gpus=1,
             is_train=True):
    ret = {}
    for lang in src_file:
        src_vocab = tf.contrib.lookup.index_table_from_file(src_vocab_file[lang], vocab_size=src_vocab_size[lang],
                                                            default_value=1)  # NOTE unk -> 1
        src_dataset = tf.data.TextLineDataset(src_file[lang])

        ## reshuffle
        if is_train == True:
            src_dataset = src_dataset.repeat(-1)
            src_dataset = src_dataset.shuffle(buffer_size=1000000)

        src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values,
                                        num_parallel_calls=10).prefetch(1000000)
        src_dataset = src_dataset.map(lambda src: src_vocab.lookup(src),
                                        num_parallel_calls=10).prefetch(1000000)
        if is_train == True:
            def key_func(src_data):
                bucket_width = (max_len + num_buckets - 1) // num_buckets
                bucket_id = tf.size(src_data) // bucket_width
                return tf.to_int64(tf.minimum(num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return windowed_data.padded_batch(batch_size_words, padded_shapes=([None]))

            def window_size_func(key):
                bucket_width = (max_len + num_buckets - 1) // num_buckets
                key += 1  # For bucket_width == 1, key 0 is unassigned.
                size = (num_gpus * batch_size_words // (key * bucket_width))
                return tf.to_int64(size)

            src_dataset = src_dataset.filter(
                lambda src: tf.logical_and(tf.size(src) <= max_len, 2 < tf.size(src)))

            src_dataset = src_dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func))
            # shuffle batch
            src_dataset = src_dataset.shuffle(buffer_size=1000)
        else:
            src_dataset = src_dataset.padded_batch(batch_size * num_gpus, padded_shapes=([None]))
        iterator = src_dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        wids = iterator.get_next()
        ret[lang] = wids
    return ret
