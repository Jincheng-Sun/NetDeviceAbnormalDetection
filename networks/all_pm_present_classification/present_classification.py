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
                               label_path='data/target',
                               out_num=53
                               )
resnet_1d = Resnet_1d()
model = Attn_model_1d(ckpt_path='models/', tsboard_path='logs/', network=resnet_1d,input_shape=[211, 1],
                   num_classes=53, feature_num=211, dev_num=33, lr=0.005, batch_size=100, regression=False)
#

# model.initialize_variables()
# model.save_tensorboard_graph()
# model.train(dataset)

#
model.restore_checkpoint(16400)
# # #
#
from sklearn.externals import joblib
target_label_en = joblib.load('models/target_label_en')
alarm_list = target_label_en.classes_




def avoid_oom(data):
    indexes = np.arange(0,data.shape[0], 10000)
    prediction = np.array([])
    for index in indexes:
        if index == indexes[-1]:
            current_batch = model.get_prediction(data[index:], is_training=False)
        else:
            current_batch = model.get_prediction(data[index:index + 10000], is_training=False)

        prediction = np.concatenate([prediction, current_batch], axis=0)
        print(index)
    return prediction

prediction = avoid_oom(dataset.test_set[:300000])
cm, accuracy = metrics_multi(
    y_pred=prediction, y_test=np.argmax(dataset.test_set['y'][:300000], axis=1), labels=alarm_list)

import matplotlib.pyplot as pyplot
pyplot.rcParams['savefig.dpi'] = 300  # pixel
pyplot.rcParams['figure.dpi'] = 300  # resolution
pyplot.rcParams["figure.figsize"] = [5,4] # figure size

draw_confusion_matrix(cm, alarm_list, precision=True, plt=pyplot)

attn1, attn2 = model.get_attn_matrix()