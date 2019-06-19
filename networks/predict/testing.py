from keras.models import load_model
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from examples.ciena.ciena_pred_dataset import pred_Dataset, pred_Dataset_2
from visualization.draw_matrix import *
import numpy as np

device_type = 'ALL'
# dataset = pred_Dataset_2(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_s_pms_3_partial_may.npy'%device_type,
#                     y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_s_alarms_2days_may.npy'%device_type)

dataset = pred_Dataset_2(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3days_may.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_may.npy'%device_type)

model = load_model('/home/oem/Projects/NetDeviceAbnormalDetection/models/predict/model_%s_s'%device_type)
pred = model.predict(dataset.test_set['x'])
results = pred
threshold = 0.8
results[results >= threshold] = 1
results[results < threshold] = 0

cm = cm_metrix(dataset.test_set['y'], results)
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
fpr = fp/(fp+ tn)
acc = (tp+tn)/(np.sum(cm))
precision = tp/(tp + fp)
recall = tp/(tp + fn)
cm_analysis(cm, ['Normal', 'malfunction'], precision=True)