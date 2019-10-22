from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_lang', type=str, default='en')
    parser.add_argument('--dev_lang', type=str, default='en')
    parser.add_argument('--test_lang', type=str, default='en')
    parser.add_argument('--is_pos', type=int, default=0, help='NER or POS?')

    parser.add_argument('--dataset', type=str, default='conll2003', help='Dataset directory')
    parser.add_argument('--dir', type=str, default=None, help='Output directory')
    parser.add_argument('--use_chars', type=int, default=1, help='Use character LSTM or not')
    parser.add_argument('--epoch', type=int, default=30)

    parser.add_argument('--emb_type', type=str, default='word', help='word | trans | word_trans')
    parser.add_argument('--emb_dim', type=int, default=300, help='Dimension of word embeddings')
    parser.add_argument('--model_dir', type=str, default='data/output_model_config_fb_wikitext103/', help='Transformer directory model')
    parser.add_argument('--layer', type=int, default=None, help='Select a single layer from Transformer')
    parser.add_argument('--trans_concat', type=str, default='all', help='all | sws | fws')
    parser.add_argument('--trans_dim', type=int, default=512, help='Transformer hidden size')
    parser.add_argument('--trans_layer', type=int, default=7, help='The total number of Transformer layers')

    parser.add_argument('--trans_type', type=str, default='monolingual', help="monolingual | crosslingual")
    parser.add_argument('--trans_vocab_src', type=str, default=None, help='Source language Transformer vocabulary')
    parser.add_argument('--trans_vocab_tgt', type=str, default=None, help='Target language Transformer vocabulary')

    args = parser.parse_args()

    # with tf.device('/cpu:0'):

    # create instance of config
    # print(args.use_attn, type(args.use_attn))

    langs = [args.train_lang, args.dev_lang, args.test_lang]
    #config = Config(mix_vocab=args.mix_vocab, use_crf=args.use_crf, mono_trans=args.mono_trans, is_pos=args.is_pos, emb_dim=args.emb_dim, src_lang=args.train_lang, tgt_lang=args.test_lang, no_glove=args.no_glove, select_layer=args.select_layer, weighted_sum_full=args.weighted_sum_full, naive_proj=args.naive_proj, highway=args.highway, weighted_sum=args.trans_weighted_sum, trans_dim=args.trans_dim, dataset=args.dataset, trans_vocab=args.trans_vocab, use_transformer=args.use_trans, dir_=args.dir, use_chars=args.use_chars, use_attn=args.use_attn, char_init=args.char_init, model_dir=args.model_dir, trans_to_output=args.trans_to_output, epoch=args.epoch)

    config = Config(args)

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter, lang=args.dev_lang)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter, lang=args.train_lang)
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter, lang=args.test_lang)

    #n_vocab = len(config.vocab_trans)
    #n_ctx = max([dev.max_seq, train.max_seq, test.max_seq])

    # with tf.device('/cpu:0'):
    # build model

    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
