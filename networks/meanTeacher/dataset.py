from networks.meanTeacher.framework.minibatching import *
import collections
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from framework.dataset import Dataset

class Dataset_classification(Dataset):
    def __init__(self, X_train_path, y_train_path, X_test_path, y_test_path):

        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        self.train_set = np.zeros(X_train.shape[0], dtype=[
            ('x', np.float32, (X_train.shape[1:])),
            ('y', np.int32, ([1]))
        ])
        self.train_set['x'] = X_train
        self.train_set['y'] = y_train

        self.test_set = np.zeros(X_test.shape[0], dtype=[
            ('x', np.float32, (X_test.shape[1:])),
            ('y', np.int32, ([1]))
        ])
        self.test_set['x'] = X_test
        self.test_set['y'] = y_test






