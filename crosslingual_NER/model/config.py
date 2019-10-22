import os
import json

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word, load_vocab_trans_multiple


class Config():
    def __init__(self, args, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load: (bool) if True, load embeddings into
                np array, else None

        """
        self.src_lang = args.train_lang
        self.tgt_lang = args.test_lang
        assert args.emb_type in ['word', 'trans', 'word_trans'], 'Embedding Type Not Supported'
        if args.emb_type == 'word':
            self.use_transformer = False
        elif args.emb_type == 'trans':
            self.use_transformer = True
            self.no_glove = True
        elif args.emb_type == 'word_trans':
            self.use_transformer = True
            self.no_glove = False

        if self.use_transformer:
            assert args.trans_type in ['monolingual', 'crosslingual'], 'Transformer Type Not Supported'
            self.trans_type = args.trans_type

            assert args.trans_concat in ['all', 'sws', 'fws'], 'Concat Type Not Supported'
            self.trans_concat = args.trans_concat

        self.model_dir = args.model_dir
        self.trans_dim = args.trans_dim
        self.trans_layer = args.trans_layer

        self.layer = args.layer
        self.is_pos = args.is_pos

        self.trans_vocab_src = args.trans_vocab_src
        self.trans_vocab_tgt = args.trans_vocab_tgt

        # general config
        self.dir_output = "results/test/"
        if args.dir:
            self.dir_output = args.dir
        self.dir_model = self.dir_output + "_model.weights/"
        self.path_log = self.dir_output + "_log.txt"

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # embeddings
        self.dim_word = args.emb_dim
        self.dim_char = 100

        self.data_dir = args.dataset

        self.filename_trimmed = "{}/glove_trimmed.npz".format(self.data_dir)
        self.use_pretrained = True

        # dataset
        self.filename_dev = "{}/dev.txt".format(self.data_dir)
        self.filename_test = "{}/test.txt".format(self.data_dir)
        self.filename_train = "{}/train.txt".format(self.data_dir)


        self.max_iter = None  # if not None, max number of examples in Dataset

        # vocab (created from dataset with build_data.py)
        self.filename_words = "{}/words.txt".format(self.data_dir)
        self.filename_tags = "{}/tags.txt".format(self.data_dir)
        self.filename_chars = "{}/chars.txt".format(self.data_dir)
       
        # training
        self.train_embeddings = False
        self.nepochs = args.epoch
        self.dropout = 0.5
        self.batch_size = 20
        self.lr_method = "adam"
        self.lr = 0.001
        self.lr_decay = 0.9
        self.clip = -1  # if negative, no clipping
        self.nepoch_no_imprv = 3

        # model hyperparameters
        self.hidden_size_char = 100  # lstm on chars
        self.hidden_size_lstm = 300  # lstm on word embeddings

        # NOTE: if both chars and crf, only 1.6x slower on GPU
        self.use_crf = True  # if crf, training is 1.7x slower on CPU
        self.use_chars = args.use_chars  # if char embedding, training is 3.5x slower on CPU

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        # self.vocab_trans = json.load(open('data/transformer_model/encoder_bpe_40000.json'))
        
        self.vocab_trans = None
        if self.use_transformer:
            vocabs_trans = {}
            assert self.trans_vocab_src is not None, "Source language vocab not provided."
            vocabs_trans[self.src_lang] = self.trans_vocab_src
            if self.trans_type == 'crosslingual':
                assert self.trans_vocab_tgt is not None, "Target language vocab not provided."
                vocabs_trans[self.tgt_lang] = self.trans_vocab_tgt
            self.vocab_trans = load_vocab_trans_multiple(vocabs_trans)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars,
                use_transformer=self.use_transformer, trans_vocab=self.vocab_trans)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)

