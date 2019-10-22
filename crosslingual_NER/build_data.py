from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors_multiple, get_processing_word
import argparse
import os


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='conll2003')
    parser.add_argument('--train_lang', type=str, default='en')
    parser.add_argument('--dev_lang', type=str, default='en')
    parser.add_argument('--test_lang', type=str, default='en')

    parser.add_argument('--src_glove', type=str, default='data/glove.42B.300d.txt')
    parser.add_argument('--tgt_glove', type=str, default=None)
    parser.add_argument('--emb_dim', type=int, default=300)

    parser.add_argument('--trimmed_glove', type=str, default='glove_trimmed.npz')

    #parser.add_argument('--init_char', type=str, default=0)
    #parser.add_argument('--trimmed_char', type=str, default='char_trimmed.npz')

    args = parser.parse_args()

    # get config and processing of words
    #config = Config(emb_dim=512, load=False, dataset='ner_nl_es', use_muse=True)
    processing_word = get_processing_word(lowercase=True)
    #src_lang = 'nl'
    #tgt_lang = 'es'

    data_dir = args.dataset

    # Generators
    dev   = CoNLLDataset(os.path.join(data_dir, 'dev.txt'), processing_word=processing_word, lang=args.dev_lang)
    test  = CoNLLDataset(os.path.join(data_dir, 'test.txt'), processing_word=processing_word, lang=args.test_lang)
    train = CoNLLDataset(os.path.join(data_dir, 'train.txt'), processing_word=processing_word, lang=args.train_lang)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])

    
    vocab_glove = get_glove_vocab(args.src_glove, lang=args.train_lang)
    if args.tgt_glove:
        vocab_glove_tgt = get_glove_vocab(args.tgt_glove, lang=args.test_lang)
        vocab = vocab_words & (vocab_glove | vocab_glove_tgt)
    else:
        vocab = vocab_words & vocab_glove
    
    #vocab = vocab_words
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, os.path.join(data_dir, 'words.txt'))
    write_vocab(vocab_tags, os.path.join(data_dir, 'tags.txt'))

    # Trim GloVe Vectors
    
    vocab = load_vocab(os.path.join(data_dir, 'words.txt'))
    if args.tgt_glove:
        gloves = {args.train_lang: args.src_glove, args.test_lang: args.tgt_glove}
    else:
        gloves = {args.train_lang: args.src_glove}
    export_trimmed_glove_vectors_multiple(vocab, gloves,
                                os.path.join(data_dir, args.trimmed_glove), args.emb_dim)
    

    # Build and save char vocab
    train = CoNLLDataset(os.path.join(data_dir, 'train.txt'))
    test  = CoNLLDataset(os.path.join(data_dir, 'test.txt'))
    dev   = CoNLLDataset(os.path.join(data_dir, 'dev.txt'))
    vocab_chars = get_char_vocab([train, test, dev])
    write_vocab(vocab_chars, os.path.join(data_dir, 'chars.txt'))

    # vocab_chars = load_vocab(os.path.join(data_dir, 'chars.txt'))
    # export_trimmed_glove_vectors(vocab_chars, config.filename_glove, config.filename_trimmed_chars, config.dim_word)


if __name__ == "__main__":
    main()
