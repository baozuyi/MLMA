
### Introduction

The code for paper Low-Resource Sequence Labeling via Unsupervised Multilingual Contextualized Representations


### deps
```
python3 
tensorflow-gpu==1.8.0
tensorflow_hub==0.1.1
```

### Uasge

The project contains two parts of codes, this dir ./ for MLMA language model, and ./crosslingual_NER for downstream sequence labeling model. The code for sequence labeling is revised based on the [code](https://github.com/guillaumegenthial/sequence_tagging?from=singlemessage).

#### For training a MLMA:

1. prepare training corpus (in the paper, we preprocess the corpus with lowercase and tokenization)

2. build vocab, run ``python data/build-vocab.py vocab_num < train_file > vocab_file`` for each corpus, for example, ``python data/build_vocab.py 200000 < data/en/en_1B_train.seg > data/en/vocab.txt`` and ``python data/build-vocab.py 200000 < data/es/eswiki_train.seg > data/es/vocab.txt``

3. change the data path in the config file
```
params["train_src"] is the path to train corpus
params["dev_src"] if you want to evaluate dev loss during the training
params["vocab_src"] is the path to the vocab file generated in Step 2
params["src_vocab_size"] is the vocab_num used in Step 2
params['init_emb'] if init from MUSE, give the path to muse embedding
```
we provide 4 template config files
```
config_emb_iden_enes.py for identical strings.
config_layer_mv_enes.py for alignment with mean and variance.
config_layer_avl_enes.py for alignment with average linkage.
config_winit_layer_avl_enes.py for average linkage + MUSE init.
```

4. run training, run ``nohup sh train.sh device config > log &``, for example, ``nohup sh train.sh 0,1,2,3 config_layer_avl_enes.py > log/log.layer_avl_enes &``. The training is time-consuming, it takes about 3 days using 4 V100. The average linkage requires large GPU memory of 16GB (when getting the OOM problem, you can reduce the vocab size or batch size to reduce the requirement for GPU memory)

when the training is done (default 40w steps), the model will be automatically dump to ``"output_model_%s"%config_file_name`` through tfhub, for example, ``config_layer_mv_enes.py to output_model_config_layer_mv_enes``


#### For training a downstream NER/POS model:

``cd crosslingual_NER``

1. prepare training data (train.txt dev.txt test.txt, like ./crosslingual_NER/ner_en_es)

2. run preprocess with build_data.py, for example, ``python build_data.py --dataset ner_en_es --train_lang en --dev_lang es --test_lang es --src_glove ../data/muse_enes/vectors-en.txt --tgt_glove ../data/muse_enes/vectors-en.txt --emb_dim 512``

3. run train and eval, two example scripts are ``run_ner_train_eval.sh run_pos_train_eval.sh``. Usage ``sh xx.sh device experiment_name data_dir mlma_model src_vocab trg_vocab``, for example, ``sh run_pos_train_eval.sh 1 pos_winit_layer_avl pos_en_es/ ../output_model_config_winit_layer_avl_enes/ ../data/en/vocab.txt ../data/es/vocab.txt > /dev/null 2>&1 &`` the code will automatically write log to results/
