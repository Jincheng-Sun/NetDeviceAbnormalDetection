import numpy as np
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from framework.dataset import Dataset
from sklearn.model_selection import train_test_split
class pr_cl_dataset(Dataset):
    def __init__(self, feature_path, dev_path, label_path):
        super().__init__()

        # train set
        X = np.load(feature_path + '_train.npy')
        dev = np.load(dev_path + '_train.npy')
        y = np.load(label_path + '_train.npy')
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, (dev.shape[1])),
            ('y', np.int32, ())
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        dev2 = np.load(dev_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy')
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, (dev2.shape[1])),
            ('y', np.int32, ())
        ])
        self.test_set['x'] = X2
        self.test_set['dev'] = dev2
        self.test_set['y'] = y2

        self.test_set, self.val_set = train_test_split(self.test_set, test_size=2/7, random_state=22)
