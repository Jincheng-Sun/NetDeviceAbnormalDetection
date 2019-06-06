import tensorflow as tf
from networks.meanTeacher.dataset import EuTk
import numpy as np
sess = tf.Session()

saver = tf.train.import_meta_graph('../../models/mean_teacher-1300.meta')
saver.restore(sess=sess, save_path='../../models/mean_teacher-1300')
graph = tf.get_default_graph()

eutk = EuTk()
test_x = eutk.test['x']
test_y = eutk.test['y']

input = graph.get_tensor_by_name('placeholders/Placeholder:0')
output = graph.get_tensor_by_name('primary/fully_connected/add:0')

# input = graph.get_tensor_by_name('placeholders/features:0')
# output = graph.get_tensor_by_name('output:0')


result = sess.run([output], feed_dict={input : test_x})[0]

result = np.argmax(result,axis=1)
result = result.reshape([-1,1])

from toolPackage.draw_cm import cm_metrix, cm_analysis
cm = cm_metrix(test_y,result)

cm_analysis(cm,['Normal', 'malfunction'],precision=True)