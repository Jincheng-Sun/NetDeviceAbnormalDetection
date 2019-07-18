import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from Models.CNN.cnn_model import CNN_model
from Models.CNN.cnn_dataset import Concat_dataset
from Networks.residual_network import Resnet_2d
from evaluation.metrics import metrics_binary, auc_roc
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
import numpy as np
threshold = 0.99
m = 5
n = 2
dataset = Concat_dataset(feature_path='data/m%s_n%s_attn_features.npy'%(m, n),
                       dev_path= 'data/m%s_n%s_attn_devices.npy'%(m, n),
                       label_path='data/m%s_n%s_attn_labels.npy'%(m, n),
                       out_num=1)
resnet_2d = Resnet_2d()

model = CNN_model(ckpt_path='models/conc_m%s'%m, tsboard_path='log/', network=resnet_2d,
                  input_shape=[m, 56, 1],num_classes=1,
                   lr = 0.001, batch_size=100, regression = True, threshold=threshold,
                  patience = 20)

model.initialize_variables()
model.save_tensorboard_graph()
model.train(dataset)

model.restore_checkpoint(984)
prediction = model.get_proba(dataset)

auc, fprs, tprs, thresholds = auc_roc(y_pred=prediction, y_test=dataset.test_set['y'])

plot_roc_curve(fprs, tprs, auc, x_axis=0.05)

cm, fpr, acc, precision, recall = metrics_binary(
    y_pred=prediction, y_test=dataset.test_set['y'],threshold=threshold)

draw_confusion_matrix(cm, ['Normal', 'Anomaly'], precision=True, font_size = 0.8)

test_dev = np.diag(np.ones([11]))
attn1, attn2 = model.get_attn_matrix(test_dev)


dev_list = ['AMP', 'ETH10G', 'ETHN', 'ETTP', 'OC192', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']

pm_list = ['BBE-RS',
            'CV-OTU', 'CV-S',
            'DROPGAINAVG-OTS', 'DROPGAINMAX-OTS_DROPGAINMIN-OTS_-',
            'E-CV', 'E-ES', 'E-INFRAMESERR_E-INFRAMES_/', 'E-OUTFRAMESERR_E-OUTFRAMES_/',
            'E-UAS', 'ES-OTU', 'ES-RS', 'ES-S',
            'OCH-OPRAVG', 'OCH-OPRMAX_OCH-OPRMIN_-', 'OCH-SPANLOSSAVG', 'OCH-SPANLOSSMAX_OCH-SPANLOSSMIN_-',
            'OPINAVG-OTS', 'OPINMAX-OTS_OPINMIN-OTS_-',
            'OPOUTAVG-OTS', 'OPOUTAVG-OTS_OPINAVG-OTS_-', 'OPOUTMAX-OTS_OPOUTMIN-OTS_-',
            'OPRAVG-OCH', 'OPRAVG-OTS', 'OPRMAX-OCH_OPRMIN-OCH_-', 'OPRMAX-OTS_OPRMIN-OTS_-',
            'OPTAVG-OCH', 'OPTAVG-OTS', 'OPTMAX-OCH_OPTMIN-OCH_-', 'OPTMAX-OTS_OPTMIN-OTS_-',
            'ORLAVG-OTS', 'ORLMIN-OTS', 'OTU-CV', 'OTU-ES', 'OTU-QAVG', 'OTU-QSTDEV',
            'PCS-CV', 'PCS-ES', 'PCS-UAS',
            'QAVG-OTU', 'QSTDEV-OTU',
            'RS-BBE', 'RS-ES',
            'S-CV', 'S-ES']


#
from networks.attention.results_process import visualize_proba
proba = visualize_proba(prediction, alarm_list=['normal', 'anomaly'])
