import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from CNN.cnn_model import CNN_model
from CNN.cnn_dataset import CNN_dataset
from Networks.residual_network_1d import Resnet_1d
from evaluation.metrics import metrics_binary, metrics_multi, auc_roc
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
import numpy as np

dataset = CNN_dataset(feature_path='data/c_PMs',
                      label_path='data/c_alm',
                      out_num = 12)
resnet_1d = Resnet_1d()
model = CNN_model(ckpt_path='models/conc', tsboard_path='log/', network=resnet_1d,
                  input_shape=[56, 1],num_classes=12,
                   lr = 0.001, batch_size=100, regression = False)
model.initialize_variables()
model.save_tensorboard_graph()
model.train(dataset)

model.restore_checkpoint(954)
prediction = model.get_prediction(dataset).reshape([-1,1])


alarm_list = ['Excessive Error Ratio', 'Frequency Out Of Range', 'GCC0 Link Failure',
              'Gauge Threshold Crossing Alert Summary', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal', 'OSC OSPF Adjacency Loss',
              'OTU Signal Degrade', 'Rx Power Out Of Range']

cm, accuracy = metrics_multi(
    y_pred=prediction, y_test=np.argmax(dataset.test_set['y'], axis=1), labels=alarm_list)

draw_confusion_matrix(cm, alarm_list, precision=True)

