import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from Models.Mean_Teacher.mean_teacher_model import Mean_Teacher_model_2d
from Models.Mean_Teacher.mean_teacher_dataset import Mean_Teacher_dataset_2d
from Networks.residual_network import Resnet_2d
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

#
dataset = Mean_Teacher_dataset_2d(feature_path='data/m3_n2_X',
                               dev_path= 'data/m3_n2_dev',
                               label_path='data/m3_n2_y',
                               unlabeled_feature_path='loose_data/m3_n2_X_un',
                               unlabeled_dev_path='loose_data/m3_n2_dev_un'
                               )
resnet_2d = Resnet_2d()
model = Mean_Teacher_model_2d(ckpt_path='models/', tsboard_path='log/', network=resnet_2d,input_shape=[3, 45, 1],
                   num_classes=1, feature_num=45, dev_num=11,
                              lr=0.0001, batch_size=1000, max_step = 20000, regression = True, patience = 50)
#
model.initialize_variables()
model.save_tensorboard_graph()
model.balance_train(dataset)
# model.train(dataset)

#
# model.restore_checkpoint(13500)
# #
# proba = model.get_proba(dataset.test_set[:20000], is_training = False)
# auc, fprs, tprs, thresholds = auc_roc(y_pred=proba, y_test=dataset.test_set['y'][:20000])
#
# plot_roc_curve(fprs, tprs, auc, x_axis=0.05)
#
# cm, fpr, acc, precision, recall = metrics_binary(
#     y_pred=proba, y_test=dataset.test_set['y'][:20000],threshold=0.95)
#
#
# import matplotlib.pyplot as pyplot
# pyplot.rcParams['savefig.dpi'] = 300  # pixel
# pyplot.rcParams['figure.dpi'] = 300  # resolution
# pyplot.rcParams["figure.figsize"] = [5,4] # figure size
#
# draw_confusion_matrix(cm, ['normal', 'anomaly'], precision=True, plt=pyplot)