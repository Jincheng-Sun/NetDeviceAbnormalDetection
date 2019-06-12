from networks.meanTeacher.model import Model
from networks.meanTeacher import minibatching
from networks.meanTeacher.dataset import EuTk, EuTk_pred
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from examples.ciena.ciena_pred_dataset import pred_Dataset
import numpy as np
import collections
model = Model()
model['rampdown_length'] = 0
model['rampup_length'] = 5000
model['training_length'] = 80000
model['max_consistency_cost'] = 50.0

tensorboard_dir = model.save_tensorboard_graph()
#
# eutk = EuTk_pred(path_x='../../data/perdevice/ETH_LossOfSignal_86_3_pm.npy',
#                  path_y='../../data/perdevice/ETH_LossOfSignal_alarm_2days.npy')
# training_batches = minibatching.training_batches(eutk.training, n_labeled_per_batch=50)
# evaluation_batches_fn = minibatching.evaluation_epoch_generator(eutk.evaluation)

device_type = 'OTM'
eutk = pred_Dataset(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3_45.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days.npy'%device_type)

training_batches = minibatching.training_batches(eutk.train_set, n_labeled_per_batch=50)
evaluation_batches_fn = minibatching.evaluation_epoch_generator(eutk.test_set)


print('finish loading dataset')


model.train(training_batches, evaluation_batches_fn)
#
# from toolPackage.draw_cm import cm_metrix, cm_analysis
# from sklearn.metrics import accuracy_score,classification_report
# test_x = eutk.test['x']
# test_y = eutk.test['y']
# model.restore_checkpoint(1000)
# result1,result2,result3 = model.test(test_x,test_y)
# result = np.argmax(result1,axis=1)
# result = result.reshape([-1,1])
# cm = cm_metrix(test_y,result)
#
# cm_analysis(cm,['Normal', 'malfunction'],precision=True)
#
# acc = accuracy_score(test_y,result)
# print(classification_report(test_y,result))