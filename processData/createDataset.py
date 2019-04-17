import numpy as np
import pandas as pd

'''load dataset'''

file_path = 'origindata'
real_alarms = pd.DataFrame(np.load('../data/%s/real_alarm_x.npy' % file_path))
real_alarm_labels = pd.DataFrame(np.load('../data/%s/real_alarm_y.npy' % file_path))
fake_normal = pd.DataFrame(np.load('../data/%s/fake_normal_x.npy' % file_path))
fake_normal_labels = pd.DataFrame(np.load('../data/%s/fake_normal_y.npy' % file_path))
real_normal = pd.DataFrame(np.load('../data/%s/real_normal_x.npy' % file_path))
real_normal_labels = pd.DataFrame(np.load('../data/%s/real_normal_y.npy' % file_path))

'''extract data'''
from sklearn.model_selection import train_test_split

_, fake_normal_x, _, fake_normal_y = train_test_split(fake_normal,
                                                      fake_normal_labels,
                                                      test_size=real_alarms.shape[0], random_state=42)
_, real_normal_x, _, real_normal_y = train_test_split(real_normal,
                                                      real_normal_labels,
                                                      test_size=real_alarms.shape[0] * 2, random_state=42)

'''concatenate'''
train_x = pd.concat([real_alarms, fake_normal_x, real_normal_x], axis=0)
train_x = pd.concat([pd.get_dummies(train_x.iloc[:,0]), train_x.iloc[:,1:46]], axis=1)
train_y = pd.concat([real_alarm_labels, fake_normal_y, real_normal_y], axis=0)

# del X_nor,x_nor,Y_nor,y_nor

# '''encode the data'''
# from keras.models import load_model
#
# encoder = load_model('../models/%s/encoder' % file_path)
# train_x = pd.DataFrame(encoder.predict(train_x))

# -------------------------------------------------------------
# '''concatenate after encoding'''
#
# train_x = np.concatenate([pd.get_dummies(train_y[1]),train_x], axis=1)  # shape = [11 + 20, -1]
# assert (train_x.shape[1] == 31)

# -------------------------------------------------------------

# '''split the dataset'''
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y[0], test_size=0.2, random_state=42)
assert (train_y.value_counts().shape == test_y.value_counts().shape)

# -------------------------------------------------------------

np.save('../data/%s/train_x.npy' % file_path, train_x)
np.save('../data/%s/test_x.npy' % file_path, test_x)

np.save('../data/%s/train_y_all.npy' % file_path, train_y)
np.save('../data/%s/test_y_all.npy' % file_path, test_y)

train_y_bi = train_y.map(lambda x: 1 if x != 'Normal' else 0)
test_y_bi = test_y.map(lambda x: 1 if x != 'Normal' else 0)

np.save('../data/%s/train_y_bi.npy' % file_path, train_y_bi)
np.save('../data/%s/test_y_bi.npy' % file_path, test_y_bi)
