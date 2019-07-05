import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from CNN.cnn_model import CNN_model
from Attention.attn_model import Attn_model
from Attention.attn_dataset import Attn_dataset
from CNN.cnn_network import Cnn_3layers
from Attention.residual_network_1d import Resnet_1d
import numpy as np

dataset = Attn_dataset(feature_path='/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_PMs',
                       dev_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_dev',
                       label_path='/home/oem/Projects/NetDeviceAbnormalDetection/data/attention/c_alm')
resnet_1d = Resnet_1d()
model = Attn_model(ckpt_path='models/attn', tsboard_path='log/', network=resnet_1d,input_shape=[45, 1],num_classes=12,
                   feature_num=45, dev_num=11, lr = 0.001, batch_size=100)
model.initialize_variables()
model.save_tensorboard_graph()
model.train(dataset)


dev_list = ['AMP', 'ETH10G', 'ETHN', 'ETTP', 'OC192', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']

alarm_list = ['Excessive Error Ratio', 'Frequency Out Of Range', 'GCC0 Link Failure',
              'Gauge Threshold Crossing Alert Summary', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal', 'OSC OSPF Adjacency Loss',
              'OTU Signal Degrade', 'Rx Power Out Of Range']


model.restore_checkpoint(1961)
prediction = model.get_prediction(dataset.test_set)
accuracy = model.get_accuracy(dataset.test_set)

test_dev = np.diag(np.ones([11]))
attn1, attn2 = model.get_attn_matrix(test_dev)

model.plot(prediction, dataset, alarm_list)