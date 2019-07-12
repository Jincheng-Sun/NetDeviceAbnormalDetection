import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from Attention.attn_model_2d import Attn_model_2d
from Attention.attn_dataset import Attn_dataset_2d
from Networks.resnet_network import Resnet
from evaluation.metrics import metrics_binary, auc_roc
from visualization.draw_matrix import draw_confusion_matrix
from visualization.draw_roc import plot_roc_curve
import numpy as np
threshold = 0.9999

# dataset = Attn_dataset_2d(feature_path='data/m3_n2_attn_features.npy',
#                        dev_path= 'data/m3_n2_attn_devices.npy',
#                        label_path='data/m3_n2_attn_labels.npy',
#                        out_num=1)
resnet = Resnet()

model = Attn_model_2d(ckpt_path='models/attn', tsboard_path='log/', network=resnet,input_shape=[3, 45, 1],
                   num_classes=1, feature_num=45, dev_num=11, lr=0.001, batch_size=100,
                   regression=True, threshold=threshold)