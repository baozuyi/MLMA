#!/usr/bin/env python
# -*- coding:utf8 -*-

from data_reader import *
import tensorflow as tf
import tensorflow_hub as tf_hub
import sys
import numpy as np
import common_attention
import common_layers
import subprocess
import re
import time
from random import shuffle
from tensorflow.python.client import device_lib
import copy

tf.logging.set_verbosity(tf.logging.INFO)

inited_emb = {}
def process_oneline(line):
    line = line.split(' ')
    try:
        return (line[0], [float(x) for x in line[1:]])
    except ValueError:
        return (line[0], [])

def prepare_encoder_input(src_wids, src_masks, lang, params, forward=None, backward=None):
    src_vocab_size = params["src_vocab_size"][lang]
    hidden_size = params["hidden_size"]
    with tf.variable_scope('WordEmbedding'):
      if params['init_emb']:
        if lang not in inited_emb:
            tf.logging.info('init embeddings ...')
            init_emb = np.random.randn(src_vocab_size, hidden_size).astype('float32') * (hidden_size ** -0.5)
            ## load vocab dict
            vocab_dict = {}
            with open(params['vocab_src'][lang], 'r') as file_in:
                for idx, line in enumerate(file_in):
                    vocab_dict[line.strip()] = idx
            ## load emb
            n_init = 0
            stime = time.time()
            with open(params['init_emb'][lang], 'r') as file_in:
                num_word, vec_len = map(int, file_in.readline().split())
                assert vec_len == hidden_size, "init emb size mismatched with hidden size"

                ## multiprocess ver @bao
                from multiprocessing import Pool
                with Pool(processes=8) as pool:
                    ret = pool.imap_unordered(process_oneline, file_in, chunksize=64)
                    for word, value in ret:
                        if len(value) != vec_len:
                            tf.logging.info('skip line with word %s and len %d'%(word, len(value)))
                            continue
                        if word in vocab_dict:
                            idx = vocab_dict[word]
                            n_init += 1
                        else:
                            idx = 1
                        init_emb[idx, :] = value
            tf.logging.info('Init %d words of %d for language %s in %fs'%(n_init, src_vocab_size, lang, time.time()-stime))
            inited_emb[lang] = init_emb

        emb_all = tf.get_variable('C_%s'%lang,
                                    initializer=inited_emb[lang],
                                    trainable=params['is_train_emb'])
      elif "mixvocab" in params and params["mixvocab"]:
        tf.logging.info(" ==== Using mixvocab for lang %s ==== "%lang) 
        emb_all = tf.get_variable('C_mix', [src_vocab_size, hidden_size], initializer=\
                      tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
      else:
        emb_all = tf.get_variable('C_%s'%lang, [src_vocab_size, hidden_size], initializer=\
                      tf.random_normal_initializer(0.0, hidden_size**-0.5, dtype=tf.float32))
      emb = tf.gather(emb_all, src_wids)

    emb *= hidden_size ** 0.5
    encoder_self_attention_bias = common_attention.attention_bias_ignore_padding(1 - src_masks)
    if forward is True:
        encoder_self_attention_bias_f = common_attention.attention_bias_lower_triangle(tf.shape(emb)[1])
    else:
        encoder_self_attention_bias_f = None
    if backward is True:
        encoder_self_attention_bias_b = common_attention.attention_bias_higher_triangle(tf.shape(emb)[1])
    else:
        encoder_self_attention_bias_b = None
    encoder_input = common_attention.add_timing_signal_1d(emb)
    encoder_input = tf.multiply(encoder_input, tf.expand_dims(src_masks, 2))
    return encoder_input, encoder_self_attention_bias, encoder_self_attention_bias_f, encoder_self_attention_bias_b


def layer_process(x, y, flag, dropout):
    if flag == None:
        return y
    for c in flag:
        if c == 'a':
            y = x + y
        elif c == 'n':
            #y = common_layers.layer_norm(y, name='layer_norm', reuse=tf.AUTO_REUSE)
            y = common_layers.layer_norm(y)
        elif c == 'd':
            y = tf.nn.dropout(y, 1.0 - dropout)
    return y


def transformer_ffn_layer(x, params):
    filter_size = params["filter_size"]
    hidden_size = params["hidden_size"]
    relu_dropout = params["relu_dropout"]
    return common_layers.conv_hidden_relu(
        x,
        filter_size,
        hidden_size,
        dropout=relu_dropout)

hidden_tensor, atten_tensor, hidden_align = [], [], []
def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        mask,
                        params=None,
                        name="encoder",
                        lang=None):
    num_hidden_layers = params["num_hidden_layers"]
    hidden_size = params["hidden_size"]
    num_heads = params["num_heads"]
    prepost_dropout = params["prepost_dropout"]
    attention_dropout = params["attention_dropout"]
    preproc_actions = params['preproc_actions']
    postproc_actions = params['postproc_actions']
    x = encoder_input
    mask = tf.expand_dims(mask, 2)
    with tf.variable_scope(name):
        for layer in range(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                ## @bao
                temp = layer_process(None, x, preproc_actions, prepost_dropout)
                hidden_tensor.append(temp)
                hidden_align.append(temp)
                o, w = common_attention.multihead_attention(
                    temp,
                    None,
                    encoder_self_attention_bias,
                    hidden_size,
                    hidden_size,
                    hidden_size,
                    num_heads,
                    attention_dropout,
                    summaries=False,
                    name="encoder_self_attention")
                x = layer_process(x, o, postproc_actions, prepost_dropout)
                o = transformer_ffn_layer(layer_process(None, x, preproc_actions, prepost_dropout), params)
                x = layer_process(x, o, postproc_actions, prepost_dropout)
                x = tf.multiply(x, mask)
                ## @bao
                atten_tensor.append(w)
        ## @bao
        temp = layer_process(None, x, preproc_actions, prepost_dropout)
        hidden_tensor.append(temp)
        hidden_align.append(temp)

        return temp

def transformer_body(body_input, params, name, lang=None):
    encoder_input, encoder_self_attention_bias, src_masks, direction_bias = body_input
    encoder_output = transformer_encoder(encoder_input, direction_bias, src_masks, params, name, lang=lang)
    return encoder_output

def output_layer(decoder_output, lang, params):
    hidden_size = params["hidden_size"]
    trg_vocab_size = params["src_vocab_size"][lang]
    with tf.variable_scope('WordEmbedding', reuse=True):
      if "mixvocab" in params and params["mixvocab"]:
        trg_emb = tf.get_variable('C_mix')
      else:
        trg_emb = tf.get_variable('C_%s'%lang)
    shape = tf.shape(decoder_output)[:-1]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, trg_emb, transpose_b=True)
    logits = tf.reshape(logits, tf.concat([shape, [trg_vocab_size]], 0))
    return logits

def sampled_softmax_layer(decoder_output, golden_label, lang, params):
    hidden_size = params["hidden_size"]
    trg_vocab_size = params["src_vocab_size"][lang]
    with tf.variable_scope('WordEmbedding', reuse=True):
      if "mixvocab" in params and params["mixvocab"]:
        trg_emb = tf.get_variable('C_mix')
      else:
        trg_emb = tf.get_variable('C_%s'%lang)

    with tf.variable_scope('SampledSoftmax', reuse=tf.AUTO_REUSE):
      m_bias = tf.get_variable('zero_bias', shape=[trg_vocab_size], trainable=False)

    shape = tf.shape(decoder_output)[:-1]
    flat_output = tf.reshape(decoder_output, [-1, hidden_size])
    flat_label = tf.reshape(golden_label, [-1, 1])

    loss = tf.nn.sampled_softmax_loss(  weights = trg_emb,
                                        biases = m_bias,
                                        labels = flat_label,
                                        inputs = flat_output,
                                        num_sampled=params['n_sampled_batch'],
                                        num_classes = trg_vocab_size,
                                        partition_strategy="div")
    loss = tf.reshape(loss, shape)
    return loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        ## Note that each grad_and_vars looks like the following:
        ##   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        #grads = []
        #for g, _ in grad_and_vars:
        #    # Add 0 dimension to the gradients to represent the tower.
        #    expanded_g = tf.expand_dims(g, 0)
        #    # Append on a 'tower' dimension which we will average over below.
        #    grads.append(expanded_g)

        ## Average over the 'tower' dimension.
        #grad = tf.concat(grads, 0)
        #grad = tf.reduce_sum(grad, 0)

        ## @bao
        grad = grad_and_vars[0][0]
        ## to dense
        if isinstance(grad, tf.IndexedSlices):
            grad = tf.convert_to_tensor(grad)
        for g, _ in grad_and_vars[1:]:
            grad += g

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def add_layer_align(loss, hidden, mask, params):
    add_loss = 0.0
    for lang in mask:
        mask[lang] = tf.reshape(mask[lang], [-1, 1])
    hidden_mean = {}
    hidden_mean_len = {}
    hidden_std = {}
    for lang in hidden:
        hidden_mean[lang] = []
        hidden_mean_len[lang] = []
        hidden_std[lang] = []
        for x in hidden[lang]:
            h = tf.reshape(x, [-1, x.get_shape()[-1]])
            hidden_mean[lang].append(tf.reduce_sum(h * mask[lang], 0) / (tf.reduce_sum(mask[lang]) + 1e-6))
            hidden_mean_len[lang].append(tf.reduce_sum(tf.abs(h * mask[lang]), 0) / (tf.reduce_sum(mask[lang]) + 1e-6))
            hidden_std[lang].append(tf.sqrt(
                                    tf.clip_by_value(
                                    tf.reduce_sum(
                                    tf.square(
                         h - tf.expand_dims(hidden_mean[lang][-1], 0)) * mask[lang], 0) / (tf.reduce_sum(mask[lang]) + 1e-6),
                                    1e-6, 1e6)))
    n_edge_alpha = 0
    n_edge_beta = 0
    n_edge_gama = 0
    langs = [lang for lang in hidden_mean]
    calculated = {lang:[] for lang in langs}
    for idx, lang_1 in enumerate(langs):
        for lang_2 in langs[idx+1:]:
            if 'layer_align_alpha' in params and params['layer_align_alpha']:
                for lang_1n, lang_2n, alpha_n, len1, len2 in zip(hidden_mean[lang_1], hidden_mean[lang_2], 
                                                              params['layer_align_alpha']*2, 
                                                              hidden_mean_len[lang_1], hidden_mean_len[lang_2]):
                    add_loss += alpha_n * tf.sqrt(tf.clip_by_value(
                                                  tf.reduce_sum(((lang_1n - lang_2n) / (len1 + len2 + 1e-6)) ** 2), 1e-6, 1e6))
                n_edge_alpha += 1

            if 'layer_align_beta' in params and params['layer_align_beta']:
                for std1, std2, beta_n in zip(hidden_std[lang_1], hidden_std[lang_2],
                                                            params['layer_align_beta']*2):
                    add_loss += beta_n * tf.sqrt(tf.clip_by_value(
                                                        tf.reduce_sum(((std1 - std2) / (std1 + std2 + 1e-6)) ** 2), 1e-6, 1e6))
                n_edge_beta += 1

            if 'layer_align_gama' in params and params['layer_align_gama']:
                tf.logging.info('== using gama type %s =='%params['layer_align_gama_type'])
                for num_layer, h1, h2, gama_n in zip(range(len(params['layer_align_gama'])*2), hidden[lang_1], 
                                                        hidden[lang_2], params['layer_align_gama']*2):
                    if gama_n == 0.0:
                        tf.logging.info('==== skip gama_n 0.0 ====')
                    if params['layer_align_gama_type'] == 'cos':
                        h1 = tf.reshape(h1, [-1, h1.get_shape()[-1]])
                        h1 = h1 / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(h1), 1)), 1)
                        h2 = tf.reshape(h2, [-1, h2.get_shape()[-1]])
                        h2 = h2 / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(h2), 1)), 1)
                        temp_mask = tf.matmul(mask[lang_1], mask[lang_2], transpose_b=True)
                        sim = tf.matmul(h1, h2, transpose_b=True)
                        h12_loss = gama_n * 0.5 * (1 - tf.reduce_sum(sim * temp_mask)/(tf.reduce_sum(temp_mask) + 1e-6))
                        if params['layer_align_gama_reg'] == True:
                            if len(calculated[lang_1]) > num_layer:
                                h11_loss = calculated[lang_1][num_layer]
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_1, 'reuse', num_layer))
                            else:
                                temp_mask1 = tf.matmul(mask[lang_1], mask[lang_1], transpose_b=True)
                                sim1 = tf.matmul(h1, h1, transpose_b=True)
                                h11_loss = gama_n * 0.5 * (1 - tf.reduce_sum(sim1 * temp_mask1)/(tf.reduce_sum(temp_mask1) + 1e-6))
                                calculated[lang_1].append(h11_loss)
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_1, 'append', num_layer))

                            if len(calculated[lang_2]) > num_layer:
                                h22_loss = calculated[lang_2][num_layer]
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_2, 'reuse', num_layer))
                            else:
                                temp_mask2 = tf.matmul(mask[lang_2], mask[lang_2], transpose_b=True)
                                sim2 = tf.matmul(h2, h2, transpose_b=True)
                                h22_loss = gama_n * 0.5 * (1 - tf.reduce_sum(sim2 * temp_mask2)/(tf.reduce_sum(temp_mask2) + 1e-6))
                                calculated[lang_2].append(h22_loss)
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_2, 'append', num_layer))

                            add_loss += (2 * h12_loss - h11_loss - h22_loss)/(len(langs)-1)
                        else:
                            add_loss += h12_loss/(len(langs)-1)
                    elif params['layer_align_gama_type'] == 'l2':
                        h1 = tf.reshape(h1, [-1, h1.get_shape()[-1]])
                        h1_sqr = tf.expand_dims(tf.reduce_sum(tf.square(h1), 1), 1)
                        h2 = tf.reshape(h2, [-1, h2.get_shape()[-1]])
                        h2_sqr = tf.expand_dims(tf.reduce_sum(tf.square(h2), 1), 0)
                        temp_mask = tf.matmul(mask[lang_1], mask[lang_2], transpose_b=True)
                        sim = tf.matmul(h1, h2, transpose_b=True)
                        h12_loss = gama_n * (tf.reduce_sum(tf.sqrt(tf.clip_by_value(h1_sqr + h2_sqr - 2 * sim, 1e-6, 1e6)) * temp_mask) / 
                                                                                              (tf.reduce_sum(temp_mask) + 1e-6))
                        if params['layer_align_gama_reg'] == True:
                            if len(calculated[lang_1]) > num_layer:
                                h11_loss = calculated[lang_1][num_layer]
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_1, 'reuse', num_layer))
                            else:
                                temp_mask1 = tf.matmul(mask[lang_1], mask[lang_1], transpose_b=True)
                                sim1 = tf.matmul(h1, h1, transpose_b=True)
                                h11_loss = gama_n * (tf.reduce_sum(tf.sqrt(tf.clip_by_value(
                                                                        h1_sqr + tf.transpose(h1_sqr) - 2 * sim1, 1e-6, 1e6)
                                                                    ) * temp_mask1) / (tf.reduce_sum(temp_mask1) + 1e-6))
                                calculated[lang_1].append(h11_loss)
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_1, 'append', num_layer))

                            if len(calculated[lang_2]) > num_layer:
                                h22_loss = calculated[lang_2][num_layer]
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_2, 'reuse', num_layer))
                            else:
                                temp_mask2 = tf.matmul(mask[lang_2], mask[lang_2], transpose_b=True)
                                sim2 = tf.matmul(h2, h2, transpose_b=True)
                                h22_loss = gama_n * (tf.reduce_sum(tf.sqrt(tf.clip_by_value(
                                                                        tf.transpose(h2_sqr) + h2_sqr - 2 * sim2, 1e-6, 1e6)
                                                                    ) * temp_mask2) / (tf.reduce_sum(temp_mask2) + 1e-6))
                                calculated[lang_2].append(h22_loss)
                                tf.logging.info("layer align gama reg %s %s %d"%(lang_2, 'append', num_layer))

                            add_loss += (2 * h12_loss - h11_loss - h22_loss)/(len(langs)-1)
                        else:
                            add_loss += h12_loss/(len(langs)-1)
                    else:
                        tf.logging.info('== InVaild layer align gama type %s =='%params['layer_align_gama_type'])
                        exit()
                n_edge_gama += 1

    tf.logging.info('==== Layer Mean Align add %d edges ===='%n_edge_alpha)
    tf.logging.info('==== Layer Var Align add %d edges ===='%n_edge_beta)
    tf.logging.info('==== Layer Sim Align add %d edges ===='%n_edge_gama)
    return {lang:loss[lang] + add_loss for lang in loss}


def add_emb_align_identical(loss, params):
    tf.logging.info('Add emb align identical loss...')
    langs = list(loss.keys())
    num_lang = len(langs)

    vocab_dict = {}
    vocab_set = {}
    for lang in langs:
        vocab_dict[lang] = {}
        vocab_set[lang] = set()
        with open(params['vocab_src'][lang], 'r') as f:
            for idx, line in enumerate(f):
                vocab_dict[lang][line.strip()] = idx
                vocab_set[lang].add(line.strip())

    for i in range(num_lang - 1):
        for j in range(i + 1, num_lang):
            src_lang, tgt_lang = langs[i], langs[j]
            add_loss = 0.0
            overlap = set.intersection(vocab_set[src_lang], vocab_set[tgt_lang])
            tf.logging.info('Find {} overlapped words between {} and {}'.format(len(overlap), src_lang, tgt_lang))
            overlap_index = {}
            for lang in [src_lang, tgt_lang]:
                overlap_index[lang] = []
            for o in overlap:
                for lang in [src_lang, tgt_lang]:
                    overlap_index[lang].append(vocab_dict[lang][o])

            emb_to_cal = []
            for lang in [src_lang, tgt_lang]:
                with tf.variable_scope('WordEmbedding', reuse=True):
                    trg_emb = tf.get_variable('C_%s'%lang)
                    emb_gather = tf.gather(trg_emb, overlap_index[lang])
                    if 'emb_align_identical_fix' in params and params['emb_align_identical_fix'] == lang:
                        emb_gather = tf.stop_gradient(emb_gather)
                        print('Stop gradient {} language'.format(lang))
                    emb_to_cal.append(emb_gather)
            diff = emb_to_cal[1] - emb_to_cal[0]
            add_loss += params['emb_align_identical_alpha'] * tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), -1)), -1) / (num_lang - 1)
            loss[src_lang] += add_loss
            loss[tgt_lang] += add_loss

    return loss

dis_acc = None
dis_loss = None
def add_layer_adv(loss, lang_hidden, lang_mask, params):
    langs = [lang for lang in lang_hidden]
    tf.logging.info('==== adding layer adv for %d languages ==='%len(langs))
    mask = {lang: tf.reshape(lang_mask[lang], [-1, ]) for lang in lang_mask}
    hidden = {}
    for lang in lang_hidden:
        temp_hidden = []
        for x in lang_hidden[lang]:
            h = tf.reshape(x, [-1, x.get_shape()[-1]])
            temp_hidden.append(h)
        hidden[lang] = tf.concat(temp_hidden, axis=1)

    xx, yy, mm = [], [], []
    for idx, lang in enumerate(langs):
        xx.append(hidden[lang])
        yy.append(tf.ones([tf.shape(hidden[lang])[0],], tf.int32) * idx)
        mm.append(mask[lang])
    xx = tf.concat(xx, axis=0)
    yy = tf.concat(yy, axis=0)
    mm = tf.concat(mm, axis=0)

    if 'layer_adv_dropout' in params and params['layer_adv_dropout']:
        tf.logging.info(" ==== Adding dropout to discriminator ==== ")
        xx = tf.nn.dropout(xx, 1.0 - params['layer_adv_dropout'])

    from flip_gradient import flip_gradient
    with tf.variable_scope('Discriminator'):
        num_gpus = params['num_gpus']
        step = tf.to_float(tf.train.get_global_step())

        pre_steps = params['layer_adv_pre_steps'] // num_gpus
        post_steps = params['layer_adv_post_steps'] // num_gpus
        alpha_inc = params['layer_adv_peak'] * tf.minimum(1.0, (step - pre_steps) / (post_steps - pre_steps))
        alpha = tf.where(step < pre_steps, 0.0, alpha_inc)

        if 'layer_adv_ad_freq' in params and params['layer_adv_ad_freq'] > 1:
            tf.logging.info(" ==== Adversarial Freq %d ===="%params['layer_adv_ad_freq'])
            alpha = tf.where(tf.equal(tf.mod(step, params['layer_adv_ad_freq']), 0.0), alpha, 0.0)

        if params['layer_adv_loss_type'] == 'flip':
            tf.logging.info(" ==== Flip Gradient ====")
            xx = flip_gradient(xx, alpha)

        if params['layer_adv_hidden'] == 0:
            W0 = tf.get_variable('W0', [xx.get_shape()[-1], len(langs)])
            b0 = tf.get_variable('b0', [len(langs),])

            oo = tf.matmul(xx, W0) + b0
        else:
            W0 = tf.get_variable('W0', [xx.get_shape()[-1], params['layer_adv_hidden']])
            b0 = tf.get_variable('b0', [params['layer_adv_hidden'],])
            W1 = tf.get_variable('W1', [params['layer_adv_hidden'], len(langs)])
            b1 = tf.get_variable('b1', [len(langs),])

            oo = tf.nn.leaky_relu(tf.matmul(xx, W0) + b0)
            if 'layer_adv_dropout' in params and params['layer_adv_dropout']:
                tf.logging.info(" ==== Adding dropout to discriminator hidden ==== ")
                oo = tf.nn.dropout(oo, 1.0 - params['layer_adv_dropout'])
            oo = tf.matmul(oo, W1) + b1

        prob = tf.nn.softmax(oo, axis=1)
        pred = tf.argmax(oo, axis=1, output_type=tf.int32)
        acc = tf.cast(tf.equal(yy, pred), dtype=tf.float32)
        acc = tf.reduce_sum(acc * mm) / (tf.reduce_sum(mm) + 1e-6)
        global dis_acc
        dis_acc = acc

        global dis_loss
        dis_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=oo, name='Dis_softmax')
        dis_loss = tf.reduce_sum(dis_loss * mm) / (tf.reduce_sum(mm) + 1e-6)

        if "layer_adv_dis_stop_acc" in params and params['layer_adv_dis_stop_acc']:
            tf.logging.info('add discriminator stop acc at %f'%params['layer_adv_dis_stop_acc'])
            dis_loss = tf.where(tf.greater(params['layer_adv_dis_stop_acc'], acc), dis_loss, 0.0)

        if params['layer_adv_loss_type'] == 'flip':
            add_loss = dis_loss
        elif params['layer_adv_loss_type'] == 'entropy':
            avg_entropy =  - tf.log(1.0 / len(langs))
            entropy = - tf.reduce_sum(prob * tf.log(tf.clip_by_value(prob, 1e-6, 1.0)), axis=1)
            add_loss = alpha * (avg_entropy - tf.reduce_mean(entropy))
        else:
            tf.logging.info('Illegal layer_adv_loss_type')
            exit()

    return {lang:loss[lang] + add_loss for lang in loss}

def get_loss(features, params, is_train=True):
    lang_hidden = {}
    lang_mask = {}

    loss = {}
    for lang, lang_feature in features.items():
        ## global variable
        hidden_tensor.clear()
        atten_tensor.clear()
        hidden_align.clear()
        # processing input
        last_padding = tf.zeros([tf.shape(lang_feature)[0], 1], tf.int64)
        src_wids = lang_feature
        src_masks = tf.to_float(tf.not_equal(src_wids, 0))

        lang_mask[lang] = src_masks

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            encoder_input, encoder_self_attention_bias, forward_bias, backward_bias = prepare_encoder_input(
                src_wids, src_masks, lang,
                params, forward=params["forward"], backward=params["backward"])

            encoder_input = tf.nn.dropout(encoder_input, 1.0 - params['prepost_dropout'])
            # preparing input and mask
            if params["forward"] is True:
                # label
                trg_wids = tf.concat([src_wids[:, 1:], last_padding], 1)

                body_input = encoder_input, encoder_self_attention_bias, src_masks, forward_bias
                body_output = transformer_body(body_input, params, name="forward", lang=lang)

                if params['sampled_softmax'] == True and is_train == True:
                    xentropy = sampled_softmax_layer(body_output, trg_wids, lang, params)
                    mask = src_masks
                else:
                    logits = output_layer(body_output, lang, params)

                    confidence = params["confidence"] if is_train else 1.0
                    trg_vocab_size = params["src_vocab_size"]

                    low_confidence = (1.0 - confidence) / tf.to_float(trg_vocab_size[lang] - 1)
                    soft_targets = tf.one_hot(tf.cast(trg_wids, tf.int32), depth=trg_vocab_size[lang],
                                                on_value=confidence, off_value=low_confidence)
                    mask = tf.cast(src_masks, logits.dtype)
                    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets)
                xentropy = xentropy * mask
                loss_f = tf.reduce_sum(xentropy) / (tf.reduce_sum(mask) + 0.01)  # forward loss

            if params["backward"] is True:

                trg_wids = tf.concat([last_padding, src_wids[:, :-1]], 1)

                body_input = encoder_input, encoder_self_attention_bias, src_masks, backward_bias
                body_output = transformer_body(body_input, params, name="backward", lang=lang)
                if params['sampled_softmax'] == True and is_train == True:
                    xentropy = sampled_softmax_layer(body_output, trg_wids, lang, params)
                    mask = src_masks
                else:
                    logits = output_layer(body_output, lang, params)
                    confidence = params["confidence"] if is_train else 1.0
                    trg_vocab_size = params["src_vocab_size"]

                    low_confidence = (1.0 - confidence) / tf.to_float(trg_vocab_size[lang] - 1)

                    soft_targets = tf.one_hot(tf.cast(trg_wids, tf.int32), depth=trg_vocab_size[lang],
                                              on_value=confidence, off_value=low_confidence)

                    mask = tf.cast(src_masks, logits.dtype)
                    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets)
                xentropy = xentropy * mask
                loss_b = tf.reduce_sum(xentropy) / (tf.reduce_sum(mask) + 0.01)

            if params["forward"] is True and params["backward"] is True:
                loss[lang] = loss_f + loss_b
            elif params["forward"] is True:
                loss[lang] = loss_f
            else:
                loss[lang] = loss_b
            ## save layer hidden
            lang_hidden[lang] = list(hidden_align)

    ## @hr
    if 'lm_loss_alpha' in params and params['lm_loss_alpha']:
        loss_alpha = params['lm_loss_alpha']
        for lang in loss_alpha:
            if lang in loss:
                tf.logging.info("Weighted LM loss for %s to %f"%(lang, loss_alpha[lang]))
                loss[lang] = loss[lang] * loss_alpha[lang]

    ## add cross-lingual regularization
    if is_train and params['layer_align']:
        loss = add_layer_align(loss, lang_hidden, lang_mask, params)
    if is_train and 'n_emb_align_identical' in params and params['n_emb_align_identical']:
        loss = add_emb_align_identical(loss, params)
    if is_train and 'layer_adv' in params and params['layer_adv']:
        loss = add_layer_adv(loss, lang_hidden, lang_mask, params)
    return loss

def transformer_export_fn(params):
    for lang in params['train_src']:
        print('========= Use %s ========='%lang)
        with tf.variable_scope('LMModel', reuse=tf.AUTO_REUSE) as var_scope:
            hidden_tensor.clear()
            atten_tensor.clear()

            hidden_align.clear()

            params_ = copy.deepcopy(params)
            for key in params_.keys():
                if "dropout" in key:
                    params_[key] = 0.0
            if 'string_input' in params and params['string_input']:
                print("Using string as placeholders")
                input_string = tf.placeholder(dtype=tf.string, shape=[None])
                input_ = input_string
                str_split = tf.sparse_tensor_to_dense(tf.string_split(input_string, ' '), default_value='<eos>')
                table = tf.contrib.lookup.index_table_from_file(vocabulary_file=params["vocab_src"][lang], default_value=1)
                features = table.lookup(str_split)

            else:
                print("Using index as placeholders")
                features = tf.placeholder(dtype=tf.int64, shape=[None, None])
                input_ = features
            loss = get_loss({lang: features}, params_, is_train=False)
            print('Hidden Tensor: ')
            for x in hidden_tensor:
                print(x)
            h = tf.concat(hidden_tensor, axis=2)
            print('Atten Tensor: ')
            for x in atten_tensor:
                print(x)
            atten = tf.concat(atten_tensor, axis=2)
            ## @bao
            tf_hub.add_signature(name=lang, inputs=input_, outputs=h)

def fill_until_num_gpus(inputs, num_gpus):
    outputs = inputs
    for i in range(num_gpus - 1):
        outputs = tf.concat([outputs, inputs], 0)
    outputs = outputs[:num_gpus, ]
    return outputs

def fill_split_feature_dict(features, num_split):
    for lang, xx in features.items():
        features[lang] = tf.cond(tf.shape(xx)[0] < num_split, 
                                    lambda: fill_until_num_gpus(xx, num_split),
                                    lambda: xx)
    ret = [{} for _ in range(num_split)]
    for lang, xx in features.items():
        feature_shards = common_layers.approximate_split(xx, num_split)
        for i in range(num_split):
            ret[i][lang] = feature_shards[i]
    return ret


def init_model(model_file):
    reader = pywrap_tensorflow.NewCheckpointReader(model_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in all_var:
        v = v.name[:-2]
        if v.endswith('_power') or v.endswith('Adam') or v.endswith('Adam_1'):
            continue
        if v in var_to_shape_map:
            tf.train.init_from_checkpoint(model_file, {v: v})

def compute_gradients_split(loss, params):
    all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    grad_emb, grad_other = [], []
    grads = tf.gradients(loss, all_var)
    for g, v in zip(grads, all_var):
        if 'Discriminator' in v.name:
            tf.logging.info('Continue Discriminator variable {}'.format(v.name))
            continue
        if 'WordEmbedding' in v.name:
            tf.logging.info('Find embedding variable {}'.format(v.name))
            grad_emb.append((g, v))
        else:
            grad_other.append((g, v))
    return grad_emb, grad_other

def split_dis_var():
    all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    dis_var, other_var = [], []
    for v in all_var:
        if 'Discriminator' in v.name:
            tf.logging.info('FInd Discriminator variable {}'.format(v.name))
            dis_var.append(v)
        else:
            other_var.append(v)
    return dis_var, other_var

def transformer_model_fn(features, mode, params):
    with tf.variable_scope('LMModel') as var_scope:
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_gpus = params['num_gpus']
            gradient_clip_value = params['gradient_clip_value']
            step = tf.to_float(tf.train.get_global_step())
            warmup_steps = params['warmup_steps'] // num_gpus
            if params['learning_rate_decay'] == 'sqrt':
                lr_warmup = params['learning_rate_peak'] * tf.minimum(1.0, step / warmup_steps)
                lr_decay = params['learning_rate_peak'] * tf.minimum(1.0, tf.sqrt(warmup_steps / step))
                lr = tf.where(step < warmup_steps, lr_warmup, lr_decay)
            elif params['learning_rate_decay'] == 'exp':
                lr = tf.train.exponential_decay(params['learning_rate_peak'],
                                                global_step=step,
                                                decay_steps=params['decay_steps'],
                                                decay_rate=params['decay_rate'])
            elif params['learning_rate_decay'] == 'none':
                lr = params['learning_rate_peak']
            else:
                tf.logging.info("learning rate decay strategy not supported")
                sys.exit()
            if params['optimizer'] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif params['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.997, epsilon=1e-09)
            elif params['optimizer'] == 'msgd':
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            elif params['optimizer'] == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
            else:
                tf.logging.info("optimizer not supported")
                sys.exit()
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, gradient_clip_value)

            if 'layer_adv' in params and params['layer_adv'] and 'layer_adv_loss_type' in params and params['layer_adv_loss_type'] == 'entropy':
                optimizer_dis = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.997, epsilon=1e-09)
                optimizer_dis = tf.contrib.estimator.clip_gradients_by_norm(optimizer_dis, gradient_clip_value)

            ## @hr
            if 'learning_rate_emb_split' in params and params['learning_rate_emb_split']:
                pre_steps = params['emb_pre_steps'] // num_gpus
                post_steps = params['emb_post_steps'] // num_gpus
                if params['learning_rate_inc_emb'] == 'linear':
                    lr_inc = params['learning_rate_peak_emb'] * tf.minimum(1.0, (step - pre_steps) / (post_steps - pre_steps))
                    lr_emb = tf.where(step < pre_steps, 0.0, lr_inc)
                else:
                    tf.logging.info("learning rate strategy for embedding not supported")
                    sys.exit()
                if params['optimizer'] == 'sgd':
                    optimizer_emb = tf.train.GradientDescentOptimizer(lr_emb)
                elif params['optimizer'] == 'adam':
                    optimizer_emb = tf.train.AdamOptimizer(learning_rate=lr_emb, beta1=0.9, beta2=0.997, epsilon=1e-09)
                elif params['optimizer'] == 'msgd':
                    optimizer_emb = tf.train.MomentumOptimizer(learning_rate=lr_emb, momentum=0.9)
                elif params['optimizer'] == 'adadelta':
                    optimizer_emb = tf.train.AdadeltaOptimizer(learning_rate=lr_emb)
                else:
                    tf.logging.info("optimizer not supported")
                    sys.exit()
                optimizer_emb = tf.contrib.estimator.clip_gradients_by_norm(optimizer_emb, gradient_clip_value)

            feature_shards = fill_split_feature_dict(features, num_gpus)
            devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
            if devices:
                loss_shards = []
                grad_shards = []
                grad_shards_emb = []
                grad_shards_other = []
                grad_shards_dis = []
                for i, device in enumerate(devices):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
                        with tf.device(device):
                            ## get_loss build the network graph
                            loss = get_loss(feature_shards[i], params)
                            loss = tf.reduce_mean([v for v in loss.values()])
                            loss_shards.append(loss)

                            dis_var, other_var = split_dis_var()
                            if 'layer_adv' in params and params['layer_adv'] and 'layer_adv_loss_type' in params and params['layer_adv_loss_type'] == 'entropy':
                                grads_dis = optimizer_dis.compute_gradients(dis_loss, dis_var)
                                grad_shards_dis.append(grads_dis)
                            ## @hr
                            if 'learning_rate_emb_split' in params and params['learning_rate_emb_split']:
                                grads_emb, grads_other = compute_gradients_split(loss, params)
                                grad_shards_emb.append(grads_emb)
                                grad_shards_other.append(grads_other)
                            else:
                                grads = optimizer.compute_gradients(loss, other_var)
                                grad_shards.append(grads)

                if 'layer_adv' in params and params['layer_adv'] and 'layer_adv_loss_type' in params and params['layer_adv_loss_type'] == 'entropy':
                    grad_dis = average_gradients(grad_shards_dis)

                loss = tf.reduce_mean(loss_shards)
                if 'learning_rate_emb_split' in params and params['learning_rate_emb_split']:
                    grad_emb = average_gradients(grad_shards_emb)
                    grad_other = average_gradients(grad_shards_other)
                else:
                    grad = average_gradients(grad_shards)
            else:
                with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                    ## get_loss build the network graph
                    loss = get_loss(feature_shards[0], params)
                    loss = tf.reduce_mean([v for v in loss.values()])

                    dis_var, other_var = split_dis_var()
                    if 'layer_adv' in params and params['layer_adv'] and 'layer_adv_loss_type' in params and params['layer_adv_loss_type'] == 'entropy':
                        grad_dis = optimizer_dis.compute_gradients(dis_loss, dis_var)

                    if 'learning_rate_emb_split' in params and params['learning_rate_emb_split']:
                        grad_emb, grad_other = compute_gradients_split(loss, params)
                    else:
                        grad = optimizer.compute_gradients(loss, other_var)

            opt_list = []
            if 'learning_rate_emb_split' in params and params['learning_rate_emb_split']:
                train_op_emb = optimizer_emb.apply_gradients(grad_emb)
                train_op_other = optimizer.apply_gradients(grad_other, global_step=tf.train.get_global_step())
                opt_list.append(train_op_emb)
                opt_list.append(train_op_other)
            else:
                train_op = optimizer.apply_gradients(grad, global_step=tf.train.get_global_step())
                opt_list.append(train_op)

            if 'layer_adv' in params and params['layer_adv'] and 'layer_adv_loss_type' in params and params['layer_adv_loss_type'] == 'entropy':
                train_op_dis = optimizer_dis.apply_gradients(grad_dis)
                opt_list.append(train_op_dis)
            train_op = tf.group(*opt_list)

            if 'load_model' in params and params['load_model'] is not None:
                print('Loading model from {}'.format(params['load_model']))
                init_model(params['load_model'])
            if 'layer_adv' in params and params['layer_adv'] and 'layer_adv_display_acc_instead_loss' in params and params['layer_adv_display_acc_instead_loss'] == True:
                return tf.estimator.EstimatorSpec(mode=mode, loss=dis_acc, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            step = tf.to_float(tf.train.get_global_step())
            tf.logging.info(step)
            params_ = copy.deepcopy(params)
            for key in params_.keys():
                if "dropout" in key:
                    params_[key] = 0.0

            num_gpus = params['num_gpus']
            feature_shards = fill_split_feature_dict(features, num_gpus)
            devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
            if devices:
                loss_shards = []
                for i, device in enumerate(devices):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                        with tf.device(device):
                            loss = get_loss(feature_shards[i], params_, is_train=False)
                            loss_shards.append(loss)
                loss = {}
                for shard in loss_shards:
                    for k, v in shard.items():
                        if k not in loss:
                            loss[k] = []
                        loss[k].append(v)
                for k in loss:
                    loss[k] = tf.reduce_mean(loss[k])
            else:
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    loss = get_loss(feature_shards[0], params_, is_train=False)
            loss = tf.reduce_mean([v for v in loss.values()])
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
