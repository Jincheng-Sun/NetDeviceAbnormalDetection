import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from CNN.cnn_model import CNN_model
from CNN.cnn_dataset import CNN_dataset
from Networks.residual_network_1d import Resnet_1d
from evaluation.metrics import metrics_binary, auc_roc
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
threshold = 0.9999
dataset = CNN_dataset(feature_path='data/c_PMs',
                      label_path='data/c_alm',
                      out_num = 1)
resnet_1d = Resnet_1d()
model = CNN_model(ckpt_path='models/conc', tsboard_path='log/', network=resnet_1d,
                  input_shape=[56, 1],num_classes=1,
                   lr = 0.001, batch_size=100, regression = True, threshold=threshold)
# model.initialize_variables()
# model.save_tensorboard_graph()
# model.train(dataset)



model.restore_checkpoint(954)
prediction = model.get_proba(dataset)

auc, fprs, tprs, thresholds = auc_roc(y_pred=prediction, y_test=dataset.test_set['y'])

plot_roc_curve(fprs, tprs, auc, x_axis=0.05)

cm, fpr, acc, precision, recall = metrics_binary(
    y_pred=prediction, y_test=dataset.test_set['y'],threshold=threshold)

draw_confusion_matrix(cm, ['Normal', 'malfunction'], precision=True)

