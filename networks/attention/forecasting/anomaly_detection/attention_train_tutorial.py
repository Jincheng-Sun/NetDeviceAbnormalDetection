import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
# Use this model from my newest repo
from Models.Attention.attn_model import Attn_model_2d
from Models.Attention.attn_model import Attn_model_2d_timeSeries
# Modify this dataset class or write your own
from Models.Attention.attn_dataset import Attn_dataset_2d
# Use this network
from Networks.residual_network import Resnet_2d
# import these evaluation and visualization lib from my repo
from evaluation.metrics import metrics_binary, auc_roc
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
import numpy as np
m = 5
n = 2
# load dataset
# For attention model, there are 3 files to load, they are generated from ./create_dataset.py
# output num is the dimension of the label, for classification it should be 12
# Please read the code of dataset class and write one in this format for classification
dataset = Attn_dataset_2d(feature_path='data/m%s_n%s_attn_features.npy'%(m, n),
                       dev_path= 'data/m%s_n%s_attn_devices.npy'%(m, n),
                       label_path='data/m%s_n%s_attn_labels.npy'%(m, n),
                       out_num=1)
# load network
resnet = Resnet_2d()

# load model
# for classification, change the num_classes to 12
model = Attn_model_2d(ckpt_path='models/attn_m%s'%m, tsboard_path='log/', network=resnet,input_shape=[m, 45, 1],
                   num_classes=1, feature_num=45, dev_num=11, lr=0.001, batch_size=100,
                   regression=True, threshold=0.99, patience=20)

# -------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------

# # initialize model and train
# model.initialize_variables()
# model.save_tensorboard_graph()
# model.train(dataset)

# -------------------------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------------------------

# Restore checkpoint using the step number
model.restore_checkpoint(1184)

# # get probability for each class
# prediction = model.get_proba(dataset.test_set)

# # calculate auc, false positive rates and etc.
# auc, fprs, tprs, thresholds = auc_roc(y_pred=prediction, y_test=dataset.test_set['y'])
#
# # plot roc curve
# plot_roc_curve(fprs, tprs, auc, x_axis=0.05)
#
# # move threshold from 0.9 to 0.9999 to see the change of fpr, acc and etc.
# threshold = 0.999
# cm, fpr, acc, precision, recall = metrics_binary(
#     y_pred=prediction, y_test=dataset.test_set['y'],threshold=threshold)
#
# # draw cm
# draw_confusion_matrix(cm, ['Normal', 'Anomaly'], precision=True, font_size = 0.8)
#
# # diagonal matrix of 11-dim representing 11 device types
# test_dev = np.diag(np.ones([11]))
#
# # output attentions, attn1 is the input attention and attn2 is the output attention
# attn1, attn2 = model.get_attn_matrix(test_dev)
#
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
#
# #
# from networks.attention.results_process import visualize_input_attention_2d
#
# # visualize attention matrix for device AMP ([0])
# attn_matrix_AMP = visualize_input_attention_2d(attn1[0], pm_list)
