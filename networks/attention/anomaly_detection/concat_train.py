import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from CNN.cnn_model import CNN_model
from CNN.cnn_dataset import Cnn_dataset
from Attention.residual_network_1d import Resnet_1d
from visualization.draw_matrix import *
from visualization.draw_roc import plot_roc_curve
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

dataset = Cnn_dataset(feature_path='data/c_PMs',
                      label_path='data/c_alm',
                      out_num = 1)
resnet_1d = Resnet_1d()
model = CNN_model(ckpt_path='models/conc', tsboard_path='log/', network=resnet_1d,input_shape=[56, 1],num_classes=1,
                   feature_num=56, dev_num=11, lr = 0.001, batch_size=100, regression = True)
# model.initialize_variables()
# model.save_tensorboard_graph()
# model.train(dataset)


dev_list = ['AMP', 'ETH10G', 'ETHN', 'ETTP', 'OC192', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']

alarm_list = ['Excessive Error Ratio', 'Frequency Out Of Range', 'GCC0 Link Failure',
              'Gauge Threshold Crossing Alert Summary', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal', 'OSC OSPF Adjacency Loss',
              'OTU Signal Degrade', 'Rx Power Out Of Range']

def metrics(threshold):
    results = model.get_logits(dataset.test_set)
    results[results >= threshold] = 1
    results[results < threshold] = 0
    cm = cm_metrix(dataset.test_set['y'], results)
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (np.sum(cm))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return cm, fpr, acc, precision, recall

def auc_roc():
    proba = model.get_logits(dataset.test_set)
    real = dataset.test_set['y']
    auc = roc_auc_score(y_true=real, y_score=proba)
    fprs, tprs, thresholds = roc_curve(y_true=real, y_score=proba)
    return auc, fprs, tprs, thresholds


model.restore_checkpoint(954)
prediction = model.get_prediction(dataset.test_set)

cm, fpr, acc, precision, recall = metrics(threshold=0.99)

cm_analysis(cm, ['Normal', 'malfunction'], precision=True)

