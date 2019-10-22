from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import argparse
import tensorflow as tf


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_lang', type=str, default='en')
    parser.add_argument('--dev_lang', type=str, default='en')
    parser.add_argument('--test_lang', type=str, default='en')
    parser.add_argument('--is_pos', type=int, default=0, help='NER or POS?')

    parser.add_argument('--dataset', type=str, default='conll2003', help='Dataset directory')
    parser.add_argument('--dir', type=str, default=None, help='Output directory')
    parser.add_argument('--use_chars', type=int, default=1, help='Use character LSTM or not')
    parser.add_argument('--char_init', type=int, default=0, help='Whether initialize char embedding')
    parser.add_argument('--epoch', type=int, default=30)

    parser.add_argument('--emb_type', type=str, default='word', help='word | trans | word_trans')
    parser.add_argument('--emb_dim', type=int, default=300, help='Dimension of word embeddings')
    parser.add_argument('--model_dir', type=str, default='data/output_model_config_fb_wikitext103/', help='Transformer directory model')
    parser.add_argument('--layer', type=int, default=None, help='Select a single layer from Transformer')
    parser.add_argument('--trans_concat', type=str, default='all', help='all | sws | fws')
    parser.add_argument('--trans_dim', type=int, default=512, help='Transformer hidden size')
    parser.add_argument('--trans_layer', type=int, default=7, help='The total number of Transformer layers')

    parser.add_argument('--trans_type', type=str, default='monolingual', help="monolingual | crosslingual")
    parser.add_argument('--trans_vocab_src', type=str, default='data/transformer_wiki_vocab_20w', help='Source language Transformer vocabulary')
    parser.add_argument('--trans_vocab_tgt', type=str, default=None, help='Target language Transformer vocabulary')


    args = parser.parse_args()

    # with tf.device('/cpu:0'):

    # create instance of config
    # print(args.use_chars, type(args.use_chars))
    langs = [args.test_lang]
    #config = Config(mix_vocab=args.mix_vocab, use_crf=args.use_crf, mono_trans=args.mono_trans, is_pos=args.is_pos, emb_dim=args.emb_dim, tgt_lang=args.test_lang, no_glove=args.no_glove, select_layer=args.select_layer, weighted_sum_full=args.weighted_sum_full, naive_proj=args.naive_proj, highway=args.highway, weighted_sum=args.trans_weighted_sum, trans_dim=args.trans_dim, dataset=args.dataset, trans_vocab=args.trans_vocab, use_transformer=args.use_trans, dir_=args.dir, use_chars=args.use_chars, use_attn=args.use_attn, char_init=args.char_init, model_dir=args.model_dir, trans_to_output=args.trans_to_output, epoch=args.epoch)

    config = Config(args)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter, lang=args.test_lang)

    # build model
    #n_vocab = len(config.vocab_trans)
    #n_ctx = test.max_seq

    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # evaluate and interact
    model.evaluate(test)
    # interactive_shell(model)


if __name__ == "__main__":
    main()
