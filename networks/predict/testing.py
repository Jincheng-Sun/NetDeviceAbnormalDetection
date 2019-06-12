from keras.models import load_model
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from examples.ciena.ciena_pred_dataset import pred_Dataset
from visualization.draw_matrix import *


device_type = 'OTM'
dataset = pred_Dataset(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3_partial_v2.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_v2.npy'%device_type)

model = load_model('modelOPTMON')
pred = model.predict(dataset.test_set['x'])
results = pred
threshold = 0.8
results[results >= threshold] = 1
results[results < threshold] = 0

cm = cm_metrix(dataset.test_set['y'], results)

cm_analysis(cm, ['Normal', 'malfunction'], precision=True)