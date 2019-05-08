import os

import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from networks.meanTeacher.utils import random_balanced_partitions, random_partitions
from networks.meanTeacher.minibatching import *


class EuTk:
    def __init__(self):
        train_x = pd.DataFrame(np.load('../../data/mean_teacher/train_x.npy'))
        train_x['x'] = train_x.apply(lambda x:x.values,axis = 1)
        train_x = train_x['x']
        train_y = pd.DataFrame(np.load('../../data/mean_teacher/train_y.npy'),columns=['y'])
        training = pd.concat([train_x, train_y], axis = 1)
        labeled = training[training['y']!=-1]
        unlabeled = training[training['y']==-1]
        lb_train,self.evaluation = train_test_split(labeled,test_size=0.2,random_state=42)
        self.training = pd.concat([lb_train, unlabeled])




