### Build Data

First, you need to put `train.txt`, `dev.txt` and `test.txt` into a directory in `data/`, e.g. `data/conll2003/`. Then, you should run `build_data.py`, with the following arguments:

```
	--dataset		directory name of your dataset [conll2003]
    --train_lang	training data language [en]
    --dev_lang		dev data language [en]
    --test_lang	test data language [en]

    --src_glove	filename of pretrained word embeddings for src lang 
    --tgt_glove	filename of pretrained word embeddings for tgt lang
    --emb_dim		dimension of pretrained embeddings [300]

    --trimmed_glove	filename of trimmed glove embeddings [glove_trimmed.npz]
```

### Train And Evaluate

Next, you can run `train.py` and `evaluate.py` with the following arguments:

```
	--train_lang		language of training data [en]
    --dev_lang			language of dev data [en]
    --test_lang		language of test data [en]
    --is_pos			1: POS 0: NER [0]

    --dataset			Dataset directory
    --dir				Output directory
    --use_chars		Use character LSTM or not [1]
    --epoch			Number of epochs [30]

    --emb_type			word:word embedding | trans:Transformer embedding | word_trans:concatenation of both [word]
    --emb_dim			Dimension of word embeddings [300]
    --model_dir		Transformer directory model
    --layer			Select a single layer from Transformer
    --trans_concat	all | sws | fws [all]
    --trans_dim		Transformer hidden size [512]
    --trans_layer		The total number of Transformer layers [7]

    --trans_type		monolingual | crosslingual [monolingual]
    --trans_vocab_src	Source language Transformer vocabulary
    --trans_vocab_tgt	Target language Transformer vocabulary
```

### Example Usage

* Monolingual NER with GloVe and Transformer embeddings

```
python build_data.py --dataset conll2003 --src_glove data/glove.42B.300d.txt
python train.py --dataset conll2003 --dir results/conll2003/ --emb_type word_trans --model_dir data/output_model_config_fb_wikitext103/ --trans_concat sws --trans_vocab_src data/transformer_wiki_vocab_20w
python evaluate.py --dataset conll2003 --dir results/conll2003/ --emb_type word_trans --model_dir data/output_model_config_fb_wikitext103/ --trans_concat sws --trans_vocab_src data/transformer_wiki_vocab_20w
```

* Cross-lingual NER using MUSE

```
python build_data.py --dataset ner_en_es --train_lang en --dev_lang es --test_lang es --src_glove data/MUSE/en-es/vectors-en.txt --tgt_glove data/MUSE/en-es/vectors-es.txt --emb_dim 512
python train.py --dataset ner_en_es --dir results/ner_en_es/ --emb_type word --train_lang en --dev_lang es --test_lang es --emb_dim 512
python evaluate.py --dataset ner_en_es --dir results/ner_en_es/ --emb_type word --train_lang en --dev_lang es --test_lang es --emb_dim 512
```

* Cross-lingual NER using Transformer embeddings

```
python build_data.py --dataset ner_en_es --train_lang en --dev_lang es --test_lang es --src_glove data/MUSE/en-es/vectors-en.txt --tgt_glove data/MUSE/en-es/vectors-es.txt --emb_dim 512
python train.py --dataset ner_en_es --dir results/ner_en_es/ --emb_type trans --train_lang en --dev_lang es --test_lang es --emb_dim 512 --model_dir data/output_model_config_fb_nalign_layer_e_e0/ --trans_concat sws --trans_type crosslingual --trans_vocab_src data/transformer_en_es_1B_vocab.20w --trans_vocab_tgt data/transformer_en_es_eswiki_vocab.20w
python evaluate.py --dataset ner_en_es --dir results/ner_en_es/ --emb_type trans --train_lang en --dev_lang es --test_lang es --emb_dim 512 --model_dir data/output_model_config_fb_nalign_layer_e_e0/ --trans_concat sws --trans_type crosslingual --trans_vocab_src data/transformer_en_es_1B_vocab.20w --trans_vocab_tgt data/transformer_en_es_eswiki_vocab.20w
```
