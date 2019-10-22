import math
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x


class Transformer:
    def __init__(self, n_vocab, n_ctx, embedding_init=None, n_embed=768, n_layer=12, n_head=12, n_transfer=12,
                 embd_pdrop=0.1, clf_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1):
        self.n_vocab = n_vocab
        self.n_embed = 300
        self.n_layer = 8
        self.n_head = 12
        self.n_ctx = n_ctx
        self.n_transfer = 12
        self.pos_init = self.position_encoding_init(self.n_ctx, self.n_embed)
        self.embedding_init = embedding_init
        self.embedding_init = np.concatenate([self.embedding_init, np.random.normal(scale=0.02, size=[self.n_ctx, self.n_embed])], 0)
        # print('num_vocab:', self.n_vocab, 'embedding_shape:', np.shape(self.embedding_init))
        # self.embd_pdrop = embd_pdrop
        # self.clf_pdrop = clf_pdrop
        # self.resid_pdrop = resid_pdrop
        # self.attn_pdrop = attn_pdrop

    def load_params(self, sess):
        shapes = json.load(open('data/transformer_model/params_shapes.json'))
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load('data/transformer_model/params_{}.npy'.format(n)) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:self.n_ctx]
        init_params[0] = np.concatenate(
            [init_params[1], init_params[0]], 0)
        del init_params[1]

        if self.n_transfer == -1:
            n_transfer = 0
        else:
            n_transfer = 1 + self.n_transfer * 12

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*transformer_model.*")
        sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])

    def gelu(self, x):
        return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))

    def shape_list(self, x):
        ps = x.get_shape().as_list()
        ts = tf.shape(x)
        return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

    def _norm(self, x, g=None, b=None, e=1e-5, axis=[1]):
        u = tf.reduce_mean(x, axis=axis, keep_dims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keep_dims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x * g + b
        return x

    def get_ema_if_exists(self, v, gvs):
        name = v.name.split(':')[0]
        ema_name = name + '/ExponentialMovingAverage:0'
        ema_v = [v for v in gvs if v.name == ema_name]
        if len(ema_v) == 0:
            ema_v = [v]
        return ema_v[0]

    def get_ema_vars(self, *vs):
        if tf.get_variable_scope().reuse:
            gvs = tf.global_variables()
            vs = [self.get_ema_if_exists(v, gvs) for v in vs]
        if len(vs) == 1:
            return vs[0]
        else:
            return vs

    def norm(self, x, scope, axis=[-1]):
        with tf.variable_scope(scope):
            n_state = self.shape_list(x)[-1]
            g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
            g, b = self.get_ema_vars(g, b)
            return self._norm(x, g, b, axis=axis)

    def dropout(self, x, pdrop, train):
        if train:
            x = tf.nn.dropout(x, 1-pdrop)
        return x

    def mask_attn_weights(self, w):
        n = self.shape_list(w)[-1]
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
        b = tf.reshape(b, [1, 1, n, n])
        w = w*b + -1e9*(1-b)
        return w

    def _attn(self, q, k, v, train=False, scale=False):
        w = tf.matmul(q, k)

        if scale:
            n_state = self.shape_list(v)[-1]
            w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

        w = self.mask_attn_weights(w)
        w = tf.nn.softmax(w)

        w = self.dropout(w, self.attn_pdrop, train)

        a = tf.matmul(w, v)
        return a

    def split_states(self, x, n):
        x_shape = self.shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1]+[n, m//n]
        return tf.reshape(x, new_x_shape)

    def merge_states(self, x):
        x_shape = self.shape_list(x)
        new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x, n, k=False):
        if k:
            return tf.transpose(self.split_states(x, n), [0, 2, 3, 1])
        else:
            return tf.transpose(self.split_states(x, n), [0, 2, 1, 3])

    def merge_heads(self, x):
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def conv1d(self, x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
        with tf.variable_scope(scope):
            nx = self.shape_list(x)[-1]
            w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
            b = tf.get_variable("b", [nf], initializer=b_init)
            if rf == 1: #faster 1x1 conv
                c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, self.shape_list(x)[:-1]+[nf])
            else: #was used to train LM
                c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
            return c

    def attn(self, x, scope, n_state, n_head, train=False, scale=False):
        # print(n_state, n_head)
        assert n_state%n_head==0
        with tf.variable_scope(scope):
            c = self.conv1d(x, 'c_attn', n_state*3, 1, train=train)
            q, k, v = tf.split(c, 3, 2)
            q = self.split_heads(q, n_head)
            k = self.split_heads(k, n_head, k=True)
            v = self.split_heads(v, n_head)
            a = self._attn(q, k, v, train=train, scale=scale)
            a = self.merge_heads(a)
            a = self.conv1d(a, 'c_proj', n_state, 1, train=train)
            a = self.dropout(a, self.resid_pdrop, train)
            return a

    def mlp(self, x, scope, n_state, train=False):
        with tf.variable_scope(scope):
            nx = self.shape_list(x)[-1]
            h = self.gelu(self.conv1d(x, 'c_fc', n_state, 1, train=train))
            h2 = self.conv1d(h, 'c_proj', nx, 1, train=train)
            h2 = self.dropout(h2, self.resid_pdrop, train)
            return h2

    def block(self, x, scope, train=False, scale=False):
        with tf.variable_scope(scope):
            nx = self.shape_list(x)[-1]
            a = self.attn(x, 'attn', nx, self.n_head, train=train, scale=scale)
            n = self.norm(x+a, 'ln_1')
            m = self.mlp(n, 'mlp', nx*4, train=train)
            h = self.norm(n+m, 'ln_2')
            return h

    def embed(self, X, we):
        we = convert_gradient_to_tensor(we)
        e = tf.gather(we, X)
        h = tf.reduce_sum(e, 2)
        return h

    def clf(self, x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
        with tf.variable_scope('clf'):
            nx = self.shape_list(x)[-1]
            w = tf.get_variable("w", [nx, ny], initializer=w_init)
            b = tf.get_variable("b", [ny], initializer=b_init)
            return tf.matmul(x, w)+b

    def position_encoding_init(self, n_position, d_pos_vec):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)] for pos in range(n_position)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        return position_enc

    def model(self, X, M, train=False, reuse=False, dropout_trans=None):
        #with tf.variable_scope('pos_we', reuse=reuse):
        #    pos_we = tf.get_variable("pos_we", [self.n_ctx, self.n_embed], initializer=tf.constant_initializer(self.pos_init))
        with tf.variable_scope('transformer_model', reuse=reuse):
            self.embd_pdrop = self.clf_pdrop = self.resid_pdrop = self.attn_pdrop = dropout_trans
            we = tf.get_variable("we", [self.n_vocab + self.n_ctx, self.n_embed], initializer=tf.constant_initializer(self.embedding_init))
            we = self.dropout(we, self.embd_pdrop, train)

            # pos_we = tf.get_variable("pos_we", [self.n_ctx, self.n_embed], initializer=tf.constant_initializer(self.pos_init))

            # X = tf.reshape(X, [-1, self.n_ctx, 2])

            h = self.embed(X, we)  # [batch, n_ctx, n_embed]

            # h = tf.nn.embedding_lookup(we, X[:, :, 0])
            # h += tf.nn.embedding_lookup(pos_we, X[:, :, 1])

            embedding = h
            for layer in range(self.n_layer):
                h = self.block(h, 'h%d'%layer, train=train, scale=True)

            lm_h = tf.reshape(h[:, :-1], [-1, self.n_embed])
            lm_logits = tf.matmul(lm_h, we, transpose_b=True)
            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
            lm_losses = tf.reshape(lm_losses, [self.shape_list(X)[0], self.shape_list(X)[1]-1])
            lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1) / (tf.reduce_sum(M[:, 1:], 1) + 0.0000000001)
            #
            # clf_h = tf.reshape(h, [-1, self.n_embed])
            # pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
            # clf_h = tf.gather(clf_h, tf.range(self.shape_list(X)[0], dtype=tf.int32)*self.n_ctx+pool_idx)
            #
            # clf_h = tf.reshape(clf_h, [-1, 2, self.n_embed])
            # if train and self.clf_pdrop > 0:
            #     shape = self.shape_list(clf_h)
            #     shape[1] = 1
            #     clf_h = tf.nn.dropout(clf_h, 1-self.clf_pdrop, shape)
            # clf_h = tf.reshape(clf_h, [-1, self.n_embed])
            # clf_logits = self.clf(clf_h, 1, train=train)
            # clf_logits = tf.reshape(clf_logits, [-1, 2])

            # clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
            return embedding, h, lm_losses  # [batch, n_ctx, n_embed]
