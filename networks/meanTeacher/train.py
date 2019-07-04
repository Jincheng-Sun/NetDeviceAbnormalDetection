from networks.meanTeacher.model import Model
from networks.meanTeacher.framework import minibatching
from networks.meanTeacher.dataset import Dataset_classification
import numpy as np

# mean-teacher + Resnet model for present tense problem
# checkpoint_path: where to save the models
# tensorboard_path: where to save the logs
# input_shape: shape of one training data
model = Model(checkpoint_path='models/', tensorboard_path='logs/', input_shape=[86,1])
#save model graph
model.save_tensorboard_graph()

# output_num: how many outputs of the network
# 2 if it's for anomaly detection and n if it's for classification.
# Here n is how many alarms we are focusing on
Model.DEFALT_HYPERPARAMS['output_num'] = 2

# X_train and X_test should be preprocessed as the flow in evaluation.ipynb and save as npy format
# to fit into the tensorflow model, X should have to be reshaped to NWC format
# which is [-1, 86, 1] before loaded into Dataset class
dataset = Dataset_classification(X_train_path='x_train.npy', y_train_path= 'y_train.npy',
                                 X_test_path='x_test.npy', y_test_path= 'y_test.npy')

# generators
training_batches = minibatching.training_batches(dataset.train_set, n_labeled_per_batch=50)
evaluation_batches_fn = minibatching.evaluation_epoch_generator(dataset.test_set)

# training, the last layer is softmax, if you want the regression output,
# change it to sigmoid or tanh and change the loss from cross entropy to mse.
model.train(training_batches, evaluation_batches_fn)
# -------------------------------------------------------------------------------------------
# testing
# -------------------------------------------------------------------------------------------

from toolPackage.draw_cm import cm_metrix, cm_analysis
from sklearn.metrics import accuracy_score,classification_report

# restore model
model.restore_checkpoint(1000)
# get prediction
prediction = model.test(dataset.test['x'])
result = np.argmax(prediction, axis=1)
result = result.reshape([-1,1])
# generate confusion matrix
cm = cm_metrix(dataset.test['y'],result)

# ['Normal', 'malfunction'] if it's anomaly detection
alarm_list = ['Excessive Error Ratio',
              'Frequency Out Of Range',
              'GCC0 Link Failure', 'Gauge Threshold Crossing Alert Summary',
              'Laser Off Far End Failure Triggered', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal',
              'OSC OSPF Adjacency Loss', 'OTU Signal Degrade',
              'Remote Fault', 'Rx Power Out Of Range']
# plot results
cm_analysis(cm, alarm_list, precision=True)

acc = accuracy_score(dataset.test['y'],result)
print(classification_report(dataset.test['y'],result))