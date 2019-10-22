#!/usr/bin/env python
# -*- coding:utf8 -*-

import os


params = {}

prefix = './data/'

params["load_model"] = None

params["train_src"] = {'en': prefix + 'en/en_1B_train.seg', 'es': prefix + 'es/eswiki_train.seg'}
params["dev_src"] = {} #{'en': prefix + 'en/en_1B_dev.seg', 'es': prefix + 'es/eswiki_dev.seg'}
params["vocab_src"] = {'en': prefix + 'en/vocab.txt', 'es': prefix + 'es/vocab.txt'}
params["src_vocab_size"] = {'en':200000, 'es':200000}
params['init_emb'] = None
params['is_train_emb'] = True # only work when init emb, should be True when init + warm up training emb

params["save_path"] = "saved_models/model_layer_avl_enes" # path for saving model

params["max_step"] = 400000
params["save_checkpoints_steps"] = 80000
params["keep_checkpoint_max"] = 5
params["max_len"] = 200
params["train_batch_size_words"] = 4096
params["optimizer"] = 'adam'
params["learning_rate_decay"] = 'none'  # sqrt: 0->peak->sqrt; exp: peak->exp, used for finetune
#params["learning_rate_decay"] = 'sqrt'  # sqrt: 0->peak->sqrt; exp: peak->exp, used for finetune
params["learning_rate_peak"] = 0.0001
params["warmup_steps"] = 12000 # only for sqrt decay
params["decay_steps"] = 100  # only for exp decay, decay every n steps
params["decay_rate"] = 0.9  # only for exp decay
params["hidden_size"] = 512
params["filter_size"] = 2048
params["num_hidden_layers"] = 6
params["num_heads"] = 8
params['gradient_clip_value'] = 5.0
params["confidence"] = 0.9  # label smoothing confidence
params["prepost_dropout"] = 0.1
params["relu_dropout"] = 0.1
params["attention_dropout"] = 0.1
params["preproc_actions"] = 'n'  # layer normalization
params["postproc_actions"] = 'da'  # dropout; residual connection

params['learning_rate_emb_split'] = False
params['learning_rate_peak_emb'] = 0.0001
params['emb_pre_steps'] = 200000
params['emb_post_steps'] = 600000
params['learning_rate_inc_emb'] = 'linear'

## l2 align
params['layer_align'] = True
params['layer_align_alpha'] = None#[0.001] * 7
params['layer_align_beta'] = None#[0.0001] * 7
params['layer_align_gama'] = [1.0] * 7
params['layer_align_gama_type'] = 'l2' # cos for cos sim, l2 for euclidean
params['layer_align_gama_reg'] = True # True for add self sim regularization

params['n_emb_align_identical'] = False
params['emb_align_identical_alpha'] = 100

## for sampled softmax
params['sampled_softmax'] = True
params['n_sampled_batch'] = 8192

params["infer_batch_size"] = 1

if not os.path.exists(params["save_path"]):
    os.mkdir(params["save_path"])

# forward, backward or both
params['forward'] = True
params['backward'] = True

params["min_start_step"]=1

