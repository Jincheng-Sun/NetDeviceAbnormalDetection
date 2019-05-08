# import collections
#
# type1 = collections.namedtuple('type1', ['attr1', 'attr2'])
#
# a = type1(attr1=1, attr2=2)
#
# print(a.attr1)
#
# #==
# Block = collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])
#
# base_depth = 128
# num_units = 5
# stride = 2
# b =Block('scope', 'bottomneck', [{
#       'depth': base_depth * 4,
#       'depth_bottleneck': base_depth,
#       'stride': 1
#   }] * (num_units - 1) + [{
#       'depth': base_depth * 4,
#       'depth_bottleneck': base_depth,
#       'stride': stride
#   }])
#
#
# for i, unit in enumerate(b.args):
#     print(i,unit)

from tensorflow.contrib import metrics, slim
import tensorflow as tf

with slim.arg_scope([slim.conv2d], kernel_size = 3):
    net = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')


    net = slim.conv2d(net, 128, scope="conv_1_1")
    net = slim.conv2d(net, 128, scope="conv_1_2")
    net = slim.conv2d(net, 128, scope="conv_1_3")
    net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
    net = slim.dropout(net, 1 - 0, scope='dropout_probability_1')

    net = slim.conv2d(net, 256, scope="conv_2_1")
    net = slim.conv2d(net, 256, scope="conv_2_2")
    net = slim.conv2d(net, 256, scope="conv_2_3")
    net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
    net = slim.dropout(net, 1 - 0, scope='dropout_probability_2')

    net = slim.conv2d(net, 512, padding='VALID', scope="conv_3_1")
    net = slim.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
    net = slim.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
    net = slim.avg_pool2d(net, [6, 6], scope='avg_pool')

    net = slim.flatten(net)
    primary_logits = slim.fully_connected(net, 10)
    secondary_logits = slim.fully_connected(net, 10)
    secondary_logits = tf.case([
        (tf.equal(num_logits, 1), lambda: primary_logits),
        (tf.equal(num_logits, 2), lambda: secondary_logits),
    ], exclusive=True, default=lambda: primary_logits)



with tf.Session() as sess:
    summary = tf.summary.FileWriter('log/',sess.graph)