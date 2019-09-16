import numpy as np
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from Models.RNN.seq2seq_model_with_attention import Seq2seq_model, Seq2seq_model_2
from Models.RNN.seq2seq_dataset import Seq2seq_dataset
from evaluation.metrics import metrics_binary, metrics_multi, auc_roc, precision_recall
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
from visualization.draw_pr import plot_pr_curve

dataset = Seq2seq_dataset('data/all_pm/m3_features.npy', 'data/all_pm/n0_targets.npy', 'data/all_pm/n0_labels.npy', 1)
# model = Seq2seq_model_2('models/all_pm/', 'logs/', 2048, 1, 211, 100, 0.001, 1, patience = 100)
# model.initialize_variables()
# model.train(dataset)


# evaluation

# model.restore_checkpoint(12400)
# result = model.get_prediction(dataset.test_set)
# real = dataset.test_set['labels']
# real = np.argmax(real, axis = 2)
#
# result_flat = result.flatten()
# real_flat = real.flatten()
# cm, acc = metrics_multi(result, real, ['normal', 'anomaly'])
#
# import matplotlib.pyplot as pyplot
# pyplot.rcParams['savefig.dpi'] = 300  # pixel
# pyplot.rcParams['figure.dpi'] = 300  # resolution
# pyplot.rcParams["figure.figsize"] = [5,4] # figure size
# draw_confusion_matrix(cm, ['normal', 'anomaly'], precision=True, plt=pyplot)

# model.restore_checkpoint(73100)
#
# proba = np.empty([0,1])
# for i in range(round(dataset.test_set.shape[0]/10000)+1):
#     proba = np.concatenate([proba, model.get_proba(dataset.test_set[i*10000:i*10000+10000], is_training = True)])
#     print(i)
#
# auc, fprs, tprs, thresholds = auc_roc(y_pred=proba, y_test=dataset.test_set['labels'])
#
# plot_roc_curve(fprs, tprs, auc, x_axis=0.05)
#
# auc, precisions, recalls, thresholds = precision_recall(y_pred=proba, y_test=dataset.test_set['labels'])
#
# plot_pr_curve(recalls, precisions, auc)
#
# cm, fpr, acc, precision, recall = metrics_binary(
#     y_pred=proba, y_test=dataset.test_set['labels'],threshold=0.9)
#
#
# import matplotlib.pyplot as pyplot
# pyplot.rcParams['savefig.dpi'] = 300  # pixel
# pyplot.rcParams['figure.dpi'] = 300  # resolution
# pyplot.rcParams["figure.figsize"] = [5,4] # figure size
#
# draw_confusion_matrix(cm, ['normal', 'anomaly'], precision=True, plt=pyplot)













