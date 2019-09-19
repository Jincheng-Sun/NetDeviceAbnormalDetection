import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from Models.Mean_Teacher.mean_teacher_model import  Mean_Teacher_model_1d
from Models.Mean_Teacher.mean_teacher_dataset import Mean_Teacher_dataset
from Networks.residual_network import Resnet_1d
from evaluation.metrics import metrics_binary, metrics_multi, auc_roc, precision_recall
from evaluation.tricks import predict_avoid_OOM
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
from visualization.draw_pr import plot_pr_curve
from networks.attention.results_process import visualize_proba
import numpy as np



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
dataset = Mean_Teacher_dataset(feature_path='data/X',
                               dev_path= 'data/dev',
                               label_path='data/y',
                               unlabeled_feature_path='data/X_un',
                               unlabeled_dev_path='data/dev_un'
                               )
resnet_1d = Resnet_1d()
model = Mean_Teacher_model_1d(ckpt_path='models/reproduce/', tsboard_path='log/', network=resnet_1d,input_shape=[45, 1],
                   num_classes=1, feature_num=45, dev_num=11, lr=0.0001, batch_size=1000, max_step = 20000, regression = True)

# model.initialize_variables()
# model.save_tensorboard_graph()
# model.train(dataset)
#
#
model.restore_checkpoint(9500)


proba = predict_avoid_OOM(model, dataset.test_set, 1)
auc, fprs, tprs, thresholds = auc_roc(y_pred=proba, y_test=dataset.test_set['y'])

plot_roc_curve(fprs, tprs, auc, x_axis=0.05)

auc, precisions, recalls, thresholds = precision_recall(y_pred=proba, y_test=dataset.test_set['y'])

plot_pr_curve(recalls, precisions, auc)

cm, fpr, acc, precision, recall = metrics_binary(
    y_pred=proba, y_test=dataset.test_set['y'],threshold=0.85)



import matplotlib.pyplot as pyplot
pyplot.rcParams['savefig.dpi'] = 300  # pixel
pyplot.rcParams['figure.dpi'] = 300  # resolution
pyplot.rcParams["figure.figsize"] = [5,4] # figure size

draw_confusion_matrix(cm, ['normal', 'anomaly'], precision=True, plt=pyplot)

attn1, attn2 = model.get_attn_matrix()