import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from Models.Mean_Teacher.mean_teacher_model import Mean_Teacher_model_1d
from networks.wrapup.present_classification.dataset import pr_cl_dataset
from Networks.residual_network import Resnet_1d
from evaluation.metrics import metrics_binary, auc_roc
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
import numpy as np

dataset = pr_cl_dataset(feature_path='data/X',
                        dev_path= 'data/dev',
                        label_path='data/y',
                        unlabeled_feature_path='data/X_un',
                        unlabeled_dev_path='data/dev_un'
                        )
resnet_1d = Resnet_1d()
model = Mean_Teacher_model_1d(ckpt_path='models/', tsboard_path='log/', network=resnet_1d,input_shape=[45, 1],
                   num_classes=12, feature_num=45, dev_num=11, lr=0.001, batch_size=100,
                   regression=False)
model.initialize_variables()
model.save_tensorboard_graph()
model.train(dataset)
