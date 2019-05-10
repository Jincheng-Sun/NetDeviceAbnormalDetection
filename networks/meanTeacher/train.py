from networks.meanTeacher.model import Model
from networks.meanTeacher import minibatching
from networks.meanTeacher.dataset import EuTk
import numpy as np
from toolPackage.draw_cm import cm_metrix, cm_analysis
from sklearn.externals import joblib
model = Model()
model['rampdown_length'] = 0
model['rampup_length'] = 5000
model['training_length'] = 40000
model['max_consistency_cost'] = 50.0

tensorboard_dir = model.save_tensorboard_graph()

eutk = EuTk()
print('finish loading dataset')
# training_batches = minibatching.training_batches(eutk.training, n_labeled_per_batch=50)
# evaluation_batches_fn = minibatching.evaluation_epoch_generator(eutk.evaluation)

# model.train(training_batches, evaluation_batches_fn)

test_x = eutk.evaluation['x']
test_y = eutk.evaluation['y']
model.restore_checkpoint(38000)
result1,result2,result3 = model.test(test_x,test_y)
result = np.argmax(result1,axis=1)
result = result.reshape([-1,1])
cm = cm_metrix(test_y,result)
labelencoder = joblib.load('../../models/labelencoder')
labels = labelencoder.classes_.tolist()
cm_analysis(cm,labels)