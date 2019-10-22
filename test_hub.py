# !/usr/bin/python
# -- coding:utf-8 --

import sys
import tensorflow as tf
import tensorflow_hub as hub

assert len(sys.argv) == 2, 'python xx.py hub_path'
m = hub.Module(sys.argv[1])
print(m)
for k, v in m.variable_map.items():
    print(k, v)
x = tf.placeholder(dtype=tf.int64, shape=[None, None])
y = m(x, signature='en') # en es ...
print(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret = sess.run(y, feed_dict={x: [[0, 0], [1, 2]]})
    print(ret)
    print(ret.shape)
