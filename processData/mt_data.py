import pandas as pd

pd.set_option('display.max_columns', 50)

import numpy as np
import sklearn.preprocessing as pre
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

scaler_path = 'Minmax_scaler'

'Europe' or 'Tokyo'
data_apostrophe = 'Europe'
# data_path = '../data/%s_Network_data.parquet' % data_apostrophe
# data_path = '../data/Eu_overlap'
data_path = '../data/Europe_Network_Data_May13.parquet'

data_path2 = '../data/Tokyo_Network_data.parquet'
data_eu = pd.read_parquet(data_path)

# data_eu2 = pd.read_parquet(data_path_new)
#data need to be reindexed since they have overlap
# data_eu = data_eu.reindex()

try:
    data_eu = data_eu.drop(['ID', 'TIME'], axis=1)
except:
    pass
try:
    data_eu = data_eu.drop(['LASTOCCURRENCE'], axis=1)
except:
    pass
#
# try:
#     data_eu2 = data_eu2.drop(['ID', 'TIME'], axis=1)
# except:
#     pass
# try:
#     data_eu2 = data_eu2.drop(['LASTOCCURRENCE'], axis=1)
# except:
#     pass

data_tk = pd.read_parquet(data_path2)

'''drop data'''
alarm = data_eu['ALARM'].value_counts()
# alm_list = alarm[(alarm > 50) & (alarm < 10000)].index.tolist()
# alm_list = alarm[alarm <= 50].index.tolist()
#
# data_eu = data_eu.drop(data_eu[data_eu['ALARM'].isin(alm_list)].index)


def drop_data(data):
    dev_list = ['CHMON', 'STM64', 'OC192', 'STTP', 'STM4', 'STM16', 'NMCMON', 'OC48', 'OC12', 'OC3', 'FLEX', 'RAMAN']

    '''drop the devices in dev_list'''
    data = data.drop(data[data['GROUPBYKEY'].isin(dev_list)].index)
    print('finish drop data')
    '''fill na with 0'''
    data = data.fillna(0)
    print('finish fill 0')
    '''data normalization'''
    try:
        scaler = joblib.load('../models/%s' % scaler_path)
        data.iloc[:, 2:47] = scaler.transform(data.iloc[:, 2:47])
        print('finish scaling')
    except:
        # should choose the right scaler
        scaler = pre.MinMaxScaler()
        data.iloc[:, 2:47] = scaler.fit_transform(data.iloc[:, 2:47])
        joblib.dump(scaler, '../models/%s' % scaler_path)

    return data


data_eu = drop_data(data_eu)
# data_eu2 = drop_data(data_eu2)
data_tk = drop_data(data_tk)

# data_eu = pd.concat([data_eu,data_eu2],axis=0)

'''Consider all the OOSs are abnormal cases'''


def in_out_service(data):
    if (data['ALARM'] == 0):
        '''OOS does not have to be malfunction, it could be simply not in use'''
        # true normal case
        if (data['LABEL'] == 'IS' or data['LABEL'] == 'n/a'):
            return 0
        else:
            # this part of the data will be dropped
            return 'Notsure'


    else:
        return data['ALARM']

'''classification'''
# data_eu['ALARM'] = data_eu.apply(in_out_service, axis=1)
# data_eu = data_eu.drop(['LABEL'], axis=1)
# data_eu = data_eu.drop(data_eu[data_eu['ALARM'] == 'Notsure'].index)
# data_tk = data_tk.drop(['LABEL'], axis=1)
# data_tk['ALARM'] = -1
#
# data = pd.concat([data_eu, data_tk], axis=0)
#
# labeled = data[data['ALARM'] != -1]
# unlabeled = data[data['ALARM'] == -1]

'''abnormal detection'''
data_eu['ALARM'] = data_eu.apply(in_out_service, axis=1)
data_eu = data_eu.drop(['LABEL'], axis=1)
data_eu = data_eu.drop(data_eu[data_eu['ALARM'] == 'Notsure'].index)
eu_normal = data_eu[data_eu['ALARM'] == 0].index
eu_alarm = data_eu[data_eu['ALARM'] != 0].index
index_unlabeled, index_normal = train_test_split(eu_normal,
                                                  test_size=eu_alarm.shape[0],
                                                  random_state=42)

data_eu['ALARM'] = -1
data_eu.loc[eu_alarm,['ALARM']] = 1
data_eu.loc[index_normal,['ALARM']] = 0

data_tk = data_tk.drop(['LABEL'], axis=1)
data_tk['ALARM'] = -1

data = pd.concat([data_eu, data_tk], axis=0)

labeled = data[data['ALARM'] != -1]
unlabeled = data[data['ALARM'] == -1]


'''encode the data'''
from keras.models import load_model

encoder_name = 'enlarge_s'
encoder = load_model('../models/encoder_%s' % encoder_name)

lbEncoder = joblib.load('../models/LE_GBK')
ohEncoder = joblib.load('../models/OH_GBK')
lbEncoder2 = joblib.load('../models/LE_ALM')
ohEncoder2 = joblib.load('../models/OH_ALM')

# if abnormal detection commit this line
# alarm = lbEncoder2.transform(np.reshape(labeled['ALARM'].tolist(), [-1, 1]))


def device_pms(data):
    pms = encoder.predict(data.iloc[:,1:46])
    device = lbEncoder.transform(np.reshape(data['GROUPBYKEY'].tolist(), [-1, 1]))
    device = np.reshape(device, [-1, 1])
    device = ohEncoder.transform(device)
    device = pd.DataFrame(device.toarray())
    x = np.concatenate([device,pms],axis=1)
    y = data['ALARM']
    return x, y

labeled = labeled.drop(labeled[labeled['GROUPBYKEY'].isin(['FLEX', 'RAMAN'])].index)
unlabeled = unlabeled.drop(unlabeled[unlabeled['GROUPBYKEY'].isin(['FLEX', 'RAMAN'])].index)

label_x, label_y = device_pms(labeled)
unlabel_x, unlabel_y = device_pms(unlabeled)
x = np.concatenate([label_x,unlabel_x],axis=0)
y = np.concatenate([label_y,unlabel_y],axis=0)
'''save data'''
np.save('../data/mean_teacher/train_x_ab_3',x)
np.save('../data/mean_teacher/train_y_ab_3',y)
