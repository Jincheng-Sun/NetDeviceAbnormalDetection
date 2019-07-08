import tensorflow as tf
from networks.meanTeacher.dataset import Dataset_classification
# checkpoint number
number = 123

# ----------------------------------------------------------------------------------
# Restore from checkpoint
# ----------------------------------------------------------------------------------
sess = tf.Session()
# load model graph
saver = tf.train.import_meta_graph('models-%s.meta'%str(number))
# load model parameters
saver.restore(sess=sess, save_path='TF_models/mean_teacher-%s'%str(number))
graph = tf.get_default_graph()

# get input and output tensors
input = graph.get_tensor_by_name('placeholders/features:0')
output = graph.get_tensor_by_name('output:0')

# load dataset
dataset = Dataset_classification(X_train_path='x_train.npy', y_train_path= 'y_train.npy',
                                 X_test_path='x_test.npy', y_test_path= 'y_test.npy')

# get prediction
prediction = sess.run(output, feed_dict={input: dataset.test_set['x']})

# ----------------------------------------------------------------------------------
# Restore from tfmodel
# ----------------------------------------------------------------------------------
from tensorflow.python.saved_model import tag_constants
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(sess=sess,tags=[tag_constants.SERVING], export_dir= 'save_path')
    input = graph.get_tensor_by_name('placeholders/features:0')
    output = graph.get_tensor_by_name('output:0')

    sess.run(output, feed_dict={input: dataset.test_set['x']})

