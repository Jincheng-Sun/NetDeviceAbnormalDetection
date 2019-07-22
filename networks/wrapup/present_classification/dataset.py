import numpy as np
from itertools import islice, chain
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from framework.dataset import Dataset
from sklearn.model_selection import train_test_split

def random_index(max_index, batch_size, random=np.random):
    def random_ranges():
        while True:
            indices = np.arange(max_index)
            random.shuffle(indices)
            yield indices

    def batch_slices(iterable):
        while True:
            yield np.array(list(islice(iterable, batch_size)))

    eternal_random_indices = chain.from_iterable(random_ranges())
    return batch_slices(eternal_random_indices)


class pr_cl_dataset(Dataset):
    def __init__(self, feature_path, dev_path, label_path,
                 unlabeled_feature_path, unlabeled_dev_path):
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

        # unlabeled train set
        X_un = np.load(unlabeled_feature_path + '_train.npy')
        dev_un = np.load(unlabeled_dev_path + '_train.npy')
        assert X_un.shape[0] == dev_un.shape[0]

        self.unlabeled_train_set = np.zeros(X_un.shape[0], dtype=[
            ('x', np.float32, (X_un.shape[1:])),
            ('dev', np.int32, (dev_un.shape[1])),
            ('y', np.int32, ())
        ])
        self.unlabeled_train_set['x'] = X_un
        self.unlabeled_train_set['dev'] = dev_un
        self.unlabeled_train_set['y'] = -1

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

    def labeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        for batch_idxs in random_index(len(self.train_set), batch_size, random):
            yield self.train_set[batch_idxs]

    def unlabeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.unlabeled_train_set) > 0
        for batch_idxs in random_index(len(self.unlabeled_train_set), batch_size, random):
            yield self.unlabeled_train_set[batch_idxs]

    def training_generator(self, batch_size=100, portion = 0.5):
        labeled = self.labeled_generator(batch_size=batch_size - int(batch_size*portion))
        unlabeled = self.unlabeled_generator(batch_size = int(batch_size*portion))
        while True:
            l = labeled.__next__()
            u = unlabeled.__next__()
            yield np.concatenate([l, u], axis=0)
