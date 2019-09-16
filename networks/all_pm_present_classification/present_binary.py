import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from networks.all_pm_present_classification.model import Attn_model_1d
from networks.all_pm_present_classification.dataset import Attn_dataset_1d
from networks.all_pm_present_classification.network import Cnn_3layers, Resnet_1d
from evaluation.metrics import metrics_binary, metrics_multi, auc_roc, precision_recall
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
from visualization.draw_pr import plot_pr_curve
from networks.attention.results_process import visualize_proba
import numpy as np

# alarm_list = ['Excessive Error Ratio', 'Frequency Out Of Range', 'GCC0 Link Failure',
#               'Gauge Threshold Crossing Alert Summary', 'Link Down', 'Local Fault',
#               'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal', 'OSC OSPF Adjacency Loss',
#               'OTU Signal Degrade', 'Rx Power Out Of Range']
#
# dev_list = ['AMP', 'ETH10G', 'ETHN', 'ETTP', 'OC192', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']
#
# pm_list = ['BBE-RS',
#             'CV-OTU', 'CV-S',
#             'DROPGAINAVG-OTS', 'DROPGAINMAX-OTS_DROPGAINMIN-OTS_-',
#             'E-CV', 'E-ES', 'E-INFRAMESERR_E-INFRAMES_/', 'E-OUTFRAMESERR_E-OUTFRAMES_/',
#             'E-UAS', 'ES-OTU', 'ES-RS', 'ES-S',
#             'OCH-OPRAVG', 'OCH-OPRMAX_OCH-OPRMIN_-', 'OCH-SPANLOSSAVG', 'OCH-SPANLOSSMAX_OCH-SPANLOSSMIN_-',
#             'OPINAVG-OTS', 'OPINMAX-OTS_OPINMIN-OTS_-',
#             'OPOUTAVG-OTS', 'OPOUTAVG-OTS_OPINAVG-OTS_-', 'OPOUTMAX-OTS_OPOUTMIN-OTS_-',
#             'OPRAVG-OCH', 'OPRAVG-OTS', 'OPRMAX-OCH_OPRMIN-OCH_-', 'OPRMAX-OTS_OPRMIN-OTS_-',
#             'OPTAVG-OCH', 'OPTAVG-OTS', 'OPTMAX-OCH_OPTMIN-OCH_-', 'OPTMAX-OTS_OPTMIN-OTS_-',
#             'ORLAVG-OTS', 'ORLMIN-OTS', 'OTU-CV', 'OTU-ES', 'OTU-QAVG', 'OTU-QSTDEV',
#             'PCS-CV', 'PCS-ES', 'PCS-UAS',
#             'QAVG-OTU', 'QSTDEV-OTU',
#             'RS-BBE', 'RS-ES',
#             'S-CV', 'S-ES']

#
dataset = Attn_dataset_1d(feature_path='data/feature',
                               dev_path= 'data/device',
                               label_path='data/target_binary',
                               out_num=1
                               )
resnet_1d = Resnet_1d()
model = Attn_model_1d(ckpt_path='models/residual/', tsboard_path='logs/', network=resnet_1d,input_shape=[211, 1],
                   num_classes=1, feature_num=211, dev_num=33, lr=0.005, batch_size=100, regression=True)
#

# model.initialize_variables()
# model.save_tensorboard_graph()
# model.train(dataset)

#
model.restore_checkpoint(17000)
# #
prediction = model.get_prediction(dataset.test_set[:5000], is_training = False)

proba = model.get_proba(dataset.test_set[:5000], is_training = False)
auc, fprs, tprs, thresholds = auc_roc(y_pred=proba, y_test=dataset.test_set['y'][:5000])

plot_roc_curve(fprs, tprs, auc, x_axis=0.05)

auc, precisions, recalls, thresholds = precision_recall(y_pred=proba, y_test=dataset.test_set['y'][:5000])

plot_pr_curve(recall=recalls, precision=precisions, auc=auc)

cm, fpr, acc, precision, recall = metrics_binary(
    y_pred=proba, y_test=dataset.test_set['y'][:5000],threshold=0.9)


import matplotlib.pyplot as pyplot
pyplot.rcParams['savefig.dpi'] = 300  # pixel
pyplot.rcParams['figure.dpi'] = 300  # resolution
pyplot.rcParams["figure.figsize"] = [5,4] # figure size

draw_confusion_matrix(cm, ['normal', 'anomaly'], precision=True, plt=pyplot)

d1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
attn1, attn2 = model.get_attn_matrix(d1)