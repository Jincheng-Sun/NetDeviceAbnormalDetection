from keras.models import load_model
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from examples.ciena.ciena_pred_dataset import pred_Dataset, pred_Dataset_2
from visualization.draw_matrix import *


device_type = 'OTM'
dataset = pred_Dataset_2(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3_partial_may.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_may.npy'%device_type)

model = load_model('model_%s'%device_type)
pred = model.predict(dataset.test_set['x'])
results = pred
threshold = 0.99
results[results >= threshold] = 1
results[results < threshold] = 0

cm = cm_metrix(dataset.test_set['y'], results)

cm_analysis(cm, ['Normal', 'malfunction'], precision=False)