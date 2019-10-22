import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub


from .data_utils import minibatches, pad_sequences, get_chunks, pad_sequences_trans, _pad_sequences
from .general_utils import Progbar
from .base_model import BaseModel


def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.trans_layer = config.layer
        self.layers = config.trans_layer
        self.trans_dim = config.trans_dim
        self.use_transformer = config.use_transformer
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def add_output(self, model, wid, lang):
        return model(wid, signature=lang)

    def add_transformer_output(self):
        model_dir = self.config.model_dir
        self.logger.info('Using transformer model from: {}'.format(model_dir))
        transformer_model = hub.Module(model_dir, trainable=self.config.train_embeddings)
        #print(transformer_model)
        #for k, v in transformer_model.variable_map.items():
        #    print(k, v)
        # x = tf.placeholder(dtype=tf.int64, shape=[None, None])
        #transformer_out = transformer_model(self.word_ids_trans, signature=self.signature)
        if self.config.trans_type == 'monolingual':
            self.logger.info('Using monolingual transformer...')
            transformer_out = transformer_model(self.word_ids_trans)
        else:
            self.logger.info('Using cross-lingual transformer...')
            transformer_out = tf.cond(tf.equal(self.signature, self.config.src_lang), 
                                  lambda: self.add_output(transformer_model, self.word_ids_trans, lang=self.config.src_lang),
                                  lambda: self.add_output(transformer_model, self.word_ids_trans, lang=self.config.tgt_lang))

        seq_len = shape_list(transformer_out)[1]
        batch_size = shape_list(transformer_out)[0]
        transformer_out = tf.reshape(transformer_out, shape=[-1, seq_len, 2, self.layers, self.trans_dim])
        transformer_out = tf.transpose(transformer_out, [0, 1, 3, 2, 4])  # [batch_size, seq_len, 7, 2, 512]
        
        if self.trans_layer is not None:
            self.layers = 1
            self.transformer_out = tf.reshape(transformer_out[:, :, self.trans_layer, :, :], [-1, seq_len, 2 * self.trans_dim])
        else:
            self.transformer_out = tf.reshape(transformer_out, [-1, seq_len, self.layers * 2 * self.trans_dim])


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        self.word_ids_trans = tf.placeholder(tf.int64, shape=[None, None],
                        name="word_ids_trans")

        self.mask_trans = tf.placeholder(tf.float32, shape=[None, None],
                                             name="mask_trans")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")
        self.dropout_trans = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_trans")

        self.signature = tf.placeholder(dtype=tf.string)

        # self.n_updates = tf.placeholder(dtype=tf.float32, shape=[], name="n_updates")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None, dropout_trans=0.1, n_updates=1, signature='en'):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            #char_ids, word_ids, word_ids_trans = zip(*words)
            if self.use_transformer:
                # word_ids_trans, sequence_lengths_trans, mask = pad_sequences_trans(word_ids, start_pos=self.n_vocab, max_length=self.n_ctx)
                char_ids, word_ids, word_ids_trans = zip(*words)
                word_ids_trans, sequence_lengths_trans = pad_sequences(word_ids_trans, 0)
            else:
                char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, word_ids_trans = zip(*words)
            if self.use_transformer:
                word_ids_trans, sequence_lengths_trans = pad_sequences(word_ids_trans, 0)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)

        # build feed dictionary

        # if self.use_transformer:
        #     sequence_lengths = sequence_lengths_trans

        feed = {
            self.sequence_lengths: sequence_lengths,
            self.word_ids: word_ids,
        }

        if self.use_transformer:
            feed[self.word_ids_trans] = word_ids_trans
            # feed[self.mask_trans] = mask

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            # if self.use_transformer:
            #     labels, _ = _pad_sequences(labels, 0, self.n_ctx)
            # else:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        # feed[self.n_updates] = n_updates
        feed[self.dropout_trans] = dropout_trans

        feed[self.signature] = signature

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """

        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")
            
            if self.config.use_transformer:
                trans_out = self.transformer_out
                if self.config.trans_concat == 'fws':
                    self.logger.info('Using 7*1024 weighted sum of transformer output...')
                    with tf.variable_scope('weighted_sum_full'):
                        
                        weighted = tf.get_variable('weights', shape=[self.layers, 2 * self.trans_dim], initializer=tf.random_normal_initializer(stddev=0.02))
                        weighted = tf.nn.softmax(weighted, 0)
                        self.weighted_input = tf.reduce_mean(weighted, -1)

                        #weighted_abs = tf.sqrt(weighted_ * weighted_)
                        #self.l1_loss = tf.reduce_sum(tf.reduce_sum(weighted_abs, -1), -1)

                        # self.weight_input = weighted
                        seq_len = shape_list(self.transformer_out)[1]
                        weighted_sum = tf.expand_dims(tf.expand_dims(weighted, 0), 0) * tf.reshape(self.transformer_out, [-1, seq_len, self.layers, 2 * self.trans_dim])
                        trans_out = tf.reduce_sum(weighted_sum, -2)

                elif self.config.trans_concat == 'sws':
                    self.logger.info('Using sws...')
                    batch = shape_list(self.transformer_out)[0]
                    seq_len = shape_list(self.transformer_out)[1]
                    with tf.variable_scope('attn_proj'):
                        w1 = tf.get_variable('w1', shape=[self.layers * 2 * self.trans_dim, 512], initializer=tf.random_normal_initializer(stddev=0.02))
                        b1 = tf.get_variable('b1', shape=[512], initializer=tf.zeros_initializer())
                        w2 = tf.get_variable('w2', shape=[512, self.layers], initializer=tf.random_normal_initializer(stddev=0.02))
                        b2 = tf.get_variable('b2', shape=[self.layers], initializer=tf.zeros_initializer())
                    transformer_out = tf.reshape(self.transformer_out, [-1, self.layers * 2 * self.trans_dim])
                    o1 = tf.tanh(tf.matmul(transformer_out, w1) + b1)
                    o2 = tf.tanh(tf.matmul(o1, w2) + b2)
                    weight = tf.nn.softmax(tf.reshape(o2, [batch, seq_len, self.layers]), -1)
                    self.weighted_input = tf.reduce_mean(tf.reduce_mean(weight, 1), 0)
                    out = tf.expand_dims(weight, -1) * tf.reshape(transformer_out, [batch, seq_len, self.layers, 2 * self.trans_dim])
                    trans_out = tf.reduce_sum(out, -2)

                if self.config.no_glove:
                    self.logger.info('Not use glove embeddings...')
                    word_embeddings = trans_out
                else:
                    word_embeddings = tf.concat([word_embeddings, trans_out], -1)
            
        with tf.variable_scope("chars"):
            if self.config.use_chars:
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            hidden_size = 2*self.config.hidden_size_lstm
                
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[hidden_size, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        #print("Before build model...")
        self.add_placeholders()

        if self.use_transformer:
            self.add_transformer_output()

        self.add_word_embeddings_op()
        self.add_logits_op()

        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        #print("After model build...")
        self.initialize_session() # now self.sess is defined and vars are init
        #print("After initialize session...")


    def predict_batch(self, words, lang):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0, dropout_trans=0, signature=lang)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        n_updates_total = nbatches * self.config.nepochs
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout, dropout_trans=0.2, n_updates=n_updates_total, signature=train.lang)
            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        if self.config.is_pos:
            return metrics["acc"]
        else:
            return metrics["f1"]

    def write_result(self, fout, labels_pred):
        for labels in labels_pred:
            for label in labels:
                fout.write(str(label) + '\n')
            fout.write('\n')

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """

        #fout = open(os.path.join(self.config.dir_output, 'tag_result.txt'), 'w')

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        type_correct_preds = {}
        type_total_correct = {}
        type_total_preds = {}
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words, test.lang)
            #self.write_result(fout, labels_pred)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                if not self.config.is_pos:
                    lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                    lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                    correct_preds += len(lab_chunks & lab_pred_chunks)
                    total_preds   += len(lab_pred_chunks)
                    total_correct += len(lab_chunks)

                    for chunk in lab_chunks:
                        t = chunk[0]
                        if t not in type_correct_preds:
                            type_correct_preds[t], type_total_correct[t], type_total_preds[t] = 0., 0., 0.
                        type_total_correct[t] += 1
                        if chunk in lab_pred_chunks:
                            type_correct_preds[t] += 1
                    for chunk in lab_pred_chunks:
                        t = chunk[0]
                        if t not in type_correct_preds:
                            type_correct_preds[t], type_total_correct[t], type_total_preds[t] = 0., 0., 0.
                        type_total_preds[t] += 1

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        d = {"acc": 100*acc, "f1": 100*f1}

        if not self.config.is_pos:
            type_p, type_r, type_f1 = {}, {}, {}
            for t in type_correct_preds:
                type_p[t] = type_correct_preds[t] / type_total_preds[t] if type_correct_preds[t] > 0 else 0
                type_r[t] = type_correct_preds[t] / type_total_correct[t] if type_correct_preds[t] > 0 else 0
                type_f1[t] = 2 * type_p[t] * type_r[t] / (type_p[t] + type_r[t]) if type_correct_preds[t] > 0 else 0
            d.update(type_f1)

        return d


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
