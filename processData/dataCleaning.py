import pandas as pd

pd.set_option('display.max_columns', 50)

import numpy as np
import sklearn.preprocessing as pre
from sklearn.externals import joblib

scaler_path = 'Minmax_scaler'

# 'Europe' or 'Tokyo'
data_apostrophe = 'Europe'
data_path = '../data/%s_Network_data.parquet' % data_apostrophe

# data_path2 = '../data/Tokyo_Network_data.parquet'
data = pd.read_parquet(data_path)
try:
    data = data.drop(['ID', 'TIME'], axis=1)
except:
    pass
# data2 = pd.read_parquet(data_path2)
# data = pd.concat([data,data2],axis=0)

# case one: Drop the fewer case, left only 11 device type (aka GROUPBYKEY) and 14 alarms

'''drop data'''

dev_list = ['CHMON', 'STM64', 'OC192', 'STTP', 'STM4', 'STM16', 'NMCMON', 'OC48', 'OC12', 'OC3']
#
label_list = ['OOS-AU', 'OOS', 'IS-ANR', 'OOS-MA', 'OOS-AUMA', 'OOS-MAANR', 'n/a']
alarm = data['ALARM'].value_counts()

alm_list = alarm[(alarm > 50) & (alarm < 10000)].index.tolist()
# alm_list = alarm[alarm < 10000].index.tolist()

'''drop the devices in dev_list'''
data = data.drop(data[data['GROUPBYKEY'].isin(dev_list)].index)
print(data['GROUPBYKEY'].value_counts())
#
'''fill na with 0'''
data = data.fillna(0)
#
'''data normalization'''
try:
    scaler = joblib.load('../models/%s' % scaler_path)
    data.iloc[:, 2:47] = scaler.transform(data.iloc[:, 2:47])

except:
    # should choose the right scaler
    scaler = pre.MinMaxScaler()
    data.iloc[:, 2:47] = scaler.fit_transform(data.iloc[:, 2:47])
    joblib.dump(scaler, '../models/%s' % scaler_path)
    print('No scaler found')
#
#
# # -------------------------------------------------------------
#
'''Consider all the OOSs are abnormal cases'''


def in_out_service(data):
    if (data['ALARM'] == 0):
        '''OOS does not have to be malfunction, it could be simply not in use'''
        # although the IS-ANR has bigger correlation with not normal
        if (data['LABEL'] == 'IS' or data['LABEL'] == 'n/a'):
            return 'Normal'
        else:
            return 'Notsure'


    else:
        return data['ALARM']


data['ALARM'] = data.apply(in_out_service, axis=1)
data = data.drop(['LABEL'], axis=1)
#
# # -------------------------------------------------------------
# modify this module, use encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

'''Embed the device type to one-hot format'''
try:
    lbEncoder = joblib.load('../models/LE_GBK')
    encoded_label = lbEncoder.transform(np.reshape(data['GROUPBYKEY'].value_counts().index.tolist(), [-1, 1]))
except:
    lbEncoder = LabelEncoder()
    encoded_label = lbEncoder.fit_transform(np.reshape(data['GROUPBYKEY'].value_counts().index.tolist(), [-1, 1]))
    joblib.dump(lbEncoder, '../models/LE_GBK')
try:
    ohEncoder = joblib.load('../models/OH_GBK')
    # onehoted = ohEncoder.transform(np.reshape(encoded_label, [-1, 1]))
except:
    ohEncoder = OneHotEncoder()
    onehoted = ohEncoder.fit_transform(np.reshape(encoded_label, [-1, 1]))
    joblib.dump(ohEncoder, '../models/OH_GBK')



'''Embed the alarm type to one-hot format'''
try:
    lbEncoder2 = joblib.load('../models/LE_ALM')
    encoded_label2 = lbEncoder2.transform(np.reshape(data['ALARM'].value_counts().index.tolist(), [-1, 1]))
except:
    lbEncoder2 = LabelEncoder()
    encoded_label2 = lbEncoder2.fit_transform(np.reshape(alm_list, [-1,1]))
    joblib.dump(lbEncoder2, '../models/LE_ALM')
try:
    ohEncoder2 = joblib.load('../models/OH_ALM')
    # onehoted = ohEncoder.transform(np.reshape(encoded_label2, [-1, 1]))
except:
    ohEncoder2 = OneHotEncoder()
    onehoted = ohEncoder2.fit_transform(np.reshape(encoded_label2, [-1, 1]))
    joblib.dump(ohEncoder2, '../models/OH_ALM')

# # -------------------------------------------------------------
#
# # '''extract data with alarms in alm_list'''
# # data with alarms that occurs 50 times more in the dataset (OOS not included)
# # real_alarms = data[data['ALARM'].isin(alm_list)]
# # data = data.drop(real_alarms.index)
# # real_alarm_labels = real_alarms['ALARM']
# # real_alarms = real_alarms.drop(['ALARM'], axis=1)
#
# # # OOS/IS-ANR data
# # fake_normal = data[data['ALARM'] == 'Malfunction']
# # data = data.drop(fake_normal.index)
# # fake_normal_labels = fake_normal['ALARM']
# # fake_normal = fake_normal.drop(['ALARM'], axis=1)
# #
# # Normal data
# # real_normal = data[data['ALARM'] == 'Normal']
# # real_normal_labels = real_normal['ALARM']
# # real_normal = real_normal.drop(['ALARM'], axis=1)
# #
# # file_path = 'preconcat'
#
# # -------------------------------------------------------------

'''extract data with alarms in alm_list'''
# data with alarms that occurs 50 times more in the dataset (OOS not included)
real_alarms = data[data['ALARM'].isin(alm_list)]
data = data.drop(real_alarms.index)
real_alarm_labels = real_alarms[['ALARM', 'GROUPBYKEY']]
real_alarms = real_alarms.drop(['ALARM', 'GROUPBYKEY'], axis=1)

# OOS/IS-ANR data
fake_normal = data[data['ALARM'] == 'Notsure']
data = data.drop(fake_normal.index)
fake_normal_labels = fake_normal[['ALARM', 'GROUPBYKEY']]
fake_normal = fake_normal.drop(['ALARM', 'GROUPBYKEY'], axis=1)

# fake_normal = pd.concat([pd.get_dummies(fake_normal_labels['GROUPBYKEY']), fake_normal], axis=1)

# Normal data
real_normal = data[data['ALARM'] == 'Normal']
del data
real_normal_labels = real_normal[['ALARM', 'GROUPBYKEY']]
real_normal = real_normal.drop(['ALARM', 'GROUPBYKEY'], axis=1)

# file_path = 'aftconcat'

# # # -------------------------------------------------------------
# #
# # '''extract data with alarms in alm_list'''
# # # data with alarms that occurs 50 times more in the dataset (OOS not included)
# # real_alarms = data[data['ALARM'].isin(alm_list)]
# # data = data.drop(real_alarms.index)
# # real_alarm_labels = real_alarms[['ALARM']]
# # real_alarms = real_alarms.drop(['ALARM'], axis=1)
# #
# # # OOS/IS-ANR data
# # fake_normal = data[data['ALARM'] == 'Malfunction']
# # data = data.drop(fake_normal.index)
# # fake_normal_labels = fake_normal[['ALARM']]
# # fake_normal = fake_normal.drop(['ALARM'], axis=1)
# #
# # # Normal data
# # real_normal = data[data['ALARM'] == 'Normal']
# # del data
# # real_normal_labels = real_normal[['ALARM']]
# # real_normal = real_normal.drop(['ALARM'], axis=1)
# #
# # file_path = 'origindata'
# #

# -------------------------------------------------------------
'''save file(all)'''

# # for clustering, classification and abnormal detection
# data['ALARM'] = data['ALARM'].astype(str)
# data.to_parquet('../data/normalized_data_%s.parquet' % data_apostrophe, engine='pyarrow')
# #
# # for training auto encoder
# np.save('../data/autoencoder/ae_data_%s' % data_apostrophe, data.drop(['LABEL', 'ALARM', 'GROUPBYKEY'], axis=1))

'''extract data'''
from sklearn.model_selection import train_test_split

# abnormal detection
_, fake_normal_x, _, fake_normal_y = train_test_split(fake_normal,
                                                      fake_normal_labels,
                                                      test_size=real_alarms.shape[0], random_state=42)
_, real_normal_x, _, real_normal_y = train_test_split(real_normal,
                                                      real_normal_labels,
                                                      test_size=real_alarms.shape[0], random_state=42)
train_x = pd.concat([real_alarms, real_normal_x], axis=0)
train_y = pd.concat([real_alarm_labels, real_normal_y], axis=0)
# # assert(train_x.shape[0] == train_y.shape[0])
# #
# # # classification
# #
# # # train_x = real_alarms
# # # train_y = real_alarm_labels
#
'''encode the data'''
from keras.models import load_model

encoder_name = 'enlarge_s'
encoder = load_model('../models/preconcat/encoder_%s' % encoder_name)
train_x = pd.DataFrame(encoder.predict(train_x))
real_alarms = pd.DataFrame(encoder.predict(real_alarms))

pd.DataFrame(train_x)
real_alarms = pd.DataFrame(real_alarms)
# #
# # # -------------------------------------------------------------
# # #
# # # # '''one-hot the device type and concatenate with the original data'''
# # # # label_encode = lbEncoder.transform(np.reshape(data['GROUPBYKEY'].tolist(), [-1,1]))
# # # # label_encode = np.reshape(label_encode, [-1,1])
# # # # oh_encode = ohEncoder.transform(label_encode)
# # # # data = pd.concat([pd.DataFrame(oh_encode.toarray()), data], axis=1)
# # # # data = data.drop(['GROUPBYKEY'], axis=1)
# # #
'''one-hot the device type and concatenate with the original data'''
label_encode = lbEncoder.transform(np.reshape(train_y['GROUPBYKEY'].tolist(), [-1, 1]))
label_encode = np.reshape(label_encode, [-1, 1])
oh_encode = ohEncoder.transform(label_encode)
oh_encode = pd.DataFrame(oh_encode.toarray())
# train_x = pd.concat([oh_encode, train_x], axis=1, ignore_index=True)
train_x = np.concatenate([oh_encode, train_x], axis=1)

'''one-hot the device type and concatenate with the original data'''
label_encode = lbEncoder.transform(np.reshape(real_alarm_labels['GROUPBYKEY'].tolist(), [-1, 1]))
label_encode = np.reshape(label_encode, [-1, 1])
oh_encode = ohEncoder.transform(label_encode)
oh_encode = pd.DataFrame(oh_encode.toarray())
# train_x = pd.concat([oh_encode, train_x], axis=1, ignore_index=True)
real_alarms = np.concatenate([oh_encode, real_alarms], axis=1)

'''one-hot the alarm type and concatenate with the original data'''
label_encode2 = lbEncoder2.transform(np.reshape(real_alarm_labels['ALARM'].tolist(), [-1, 1]))
label_encode2 = np.reshape(label_encode2, [-1, 1])
oh_encode2 = ohEncoder2.transform(label_encode2)
y_all_train = pd.DataFrame(oh_encode2.toarray())
# train_x = pd.concat([oh_encode, train_x], axis=1, ignore_index=True)


# # # assert(train_x.shape[0] == 2*real_alarms.shape[0])
# # #
# # # # -------------------------------------------------------------
train_y = train_y['ALARM'].map(lambda x: 1 if x != 'Normal' else 0)
np.save('../data/new/%s_train_x_bi.npy' % encoder_name, train_x)
np.save('../data/new/%s_train_y_bi.npy' % encoder_name, train_y)
np.save('../data/new/%s_train_x_all.npy' % encoder_name, real_alarms)
np.save('../data/new/%s_train_y_all.npy' % encoder_name, y_all_train)
# # # -------------------------------------------------------------
# #
# #
# #
# # np.save('../data/new/train_x_all.npy', train_x)
# # np.save('../data/new/train_y_all.npy', train_y)
# # #
# # # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y[0], test_size=0.2, random_state=42)
# # # assert (train_y.value_counts().shape == test_y.value_counts().shape)
# # #
