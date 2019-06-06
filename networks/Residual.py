import sys
sys.path.append('/home/oem/Projects')
from Kylearn.resnet import Resnet
import tensorflow as tf

resnet = Resnet()

with tf.name_scope('input'):
    features = tf.get_variable('input_x', shape=[None,3, 86, 1], dtype=tf.float32)
    label = tf.get_variable('label', shape=[None], dtype=tf.float32)

    