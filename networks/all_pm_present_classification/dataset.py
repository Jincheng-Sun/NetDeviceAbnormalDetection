import numpy as np
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.mini_batch import random_index
import collections
class Attn_dataset_1d(Dataset):
    def __init__(self, feature_path, dev_path, label_path, out_num):
        super().__init__()

        # train set
        X = np.load(feature_path + '_train.npy')
        X = np.expand_dims(X, -1)
        dev = np.load(dev_path + '_train.npy')
        # y = np.load(label_path + '_train.npy').reshape(-1)
        # y = y-1
        # y = np.eye(out_num)[y].reshape([-1, out_num])
        y = np.load(label_path + '_train.npy')
        y = y.reshape([-1, out_num])
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, (dev.shape[1])),
            ('y', np.int32, ([out_num]))
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        X2 = np.expand_dims(X2, -1)
        dev2 = np.load(dev_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy').reshape([-1,out_num])
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, (dev2.shape[1])),
            ('y', np.int32, ([out_num]))
        ])
        self.test_set['x'] = X2
        self.test_set['dev'] = dev2
        self.test_set['y'] = y2

        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.002, random_state=22)

    # def labeled_pos_generator(self, batch_size=50, random=np.random):
    #     assert batch_size > 0 and len(self.train_set) > 0
    #     anomaly = self.train_set[self.train_set['y'].flatten() == 1]
    #     for batch_idxs in random_index(len(anomaly), batch_size, random):
    #         yield anomaly[batch_idxs]
    #
    # def labeled_neg_generator(self, batch_size=50, random=np.random):
    #     assert batch_size > 0 and len(self.train_set) > 0
    #     normal = self.train_set[self.train_set['y'].flatten() == 0]
    #     for batch_idxs in random_index(len(normal), batch_size, random):
    #         yield normal[batch_idxs]
    #
    # def training_generator(self, batch_size=100, pos=500, neg=500):
    #     labeled_pos = self.labeled_pos_generator(batch_size=pos)
    #     labeled_neg = self.labeled_neg_generator(batch_size=neg)
    #     return zip(labeled_pos, labeled_neg)