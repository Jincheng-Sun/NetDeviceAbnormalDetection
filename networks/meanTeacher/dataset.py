import os

import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from networks.meanTeacher.utils import random_balanced_partitions, random_partitions
from networks.meanTeacher.minibatching import *


class EuTk:
    def __init__(self):
        train_x = np.load('../../data/mean_teacher/train_x.npy')
        train_y = np.load('../../data/mean_teacher/train_y.npy')
        train_x = train_x.reshape([-1, 86, 1])
        array = np.zeros(train_x.shape[0], dtype=[
            ('x', np.float32, (86, 1)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        array['x'] = train_x
        array['y'] = train_y
        labeled = array[array['y']!=-1]
        unlabeled = array[array['y']==-1]
        lb_train,self.evaluation = train_test_split(labeled,test_size=0.2,random_state=42)
        self.training = np.concatenate([lb_train, unlabeled],axis=0)




