import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from Models.Attention.attn_model import Attn_model_1d
from Models.Attention.attn_dataset import Attn_dataset
from Networks.residual_network import Resnet_1d
from evaluation.metrics import metrics_binary, metrics_multi, auc_roc
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
from networks.attention.results_process import visualize_proba
import numpy as np

alarm_list = ['Excessive Error Ratio', 'Frequency Out Of Range', 'GCC0 Link Failure',
              'Gauge Threshold Crossing Alert Summary', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal', 'OSC OSPF Adjacency Loss',
              'OTU Signal Degrade', 'Rx Power Out Of Range']

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

dataset = Attn_dataset(feature_path='/home/oem/Projects/NetDeviceAbnormalDetection/networks/wrapup/present_classification/data/X',
                       dev_path= '/home/oem/Projects/NetDeviceAbnormalDetection/networks/wrapup/present_classification/data/dev',
                       label_path='/home/oem/Projects/NetDeviceAbnormalDetection/networks/wrapup/present_classification/data/y',
                       out_num=12)
resnet_1d = Resnet_1d()
model = Attn_model_1d(ckpt_path='models/attn', tsboard_path='log/', network=resnet_1d,input_shape=[45, 1],
                   num_classes=12, feature_num=45, dev_num=11, lr=0.001, batch_size=100,
                   regression=False)
model.initialize_variables()
model.save_tensorboard_graph()
model.train(dataset)

# model.restore_checkpoint(6731)

# -------------------------------------------------------------------------------------------
# overall
# -------------------------------------------------------------------------------------------

prediction = model.get_prediction(dataset, is_training = True).reshape([-1,1])

cm, accuracy = metrics_multi(
    y_pred=prediction, y_test=np.argmax(dataset.test_set['y'], axis=1), labels=alarm_list)

draw_confusion_matrix(cm, alarm_list, precision=True)

# # -------------------------------------------------------------------------------------------
# # per class
# # -------------------------------------------------------------------------------------------
#
# # threshold = 0.9
# index = 0
#
# proba = model.get_proba(dataset)
# real = dataset.test_set['y']
#
# proba_i = proba[:, index]
# real_i = real[:, index]
# import collections
# print(collections.Counter(real_i))
# auc, fprs, tprs, thresholds = auc_roc(y_pred=proba_i, y_test=real_i)
#
# plot_roc_curve(fprs, tprs, auc, x_axis=1)
#
# proba = visualize_proba(proba, alarm_list)
#
# # cm, fpr, acc, precision, recall = metrics_binary(
# #     y_pred=proba_i, y_test=real_i, threshold=threshold)
#
# # draw_confusion_matrix(cm, ['Normal', 'malfunction'], precision=True)


# -------------------------------------------------------------------------------------------
# attn matrix
# -------------------------------------------------------------------------------------------

test_dev = np.diag(np.ones([11]))
attn1, attn2 = model.get_attn_matrix(test_dev)





#
from networks.attention.results_process import visualize_input_attention, visualize_output_attention
attn1 = visualize_input_attention(attn1, dev_list, pm_list)
attn2 = visualize_output_attention(attn2, dev_list, alarm_list)