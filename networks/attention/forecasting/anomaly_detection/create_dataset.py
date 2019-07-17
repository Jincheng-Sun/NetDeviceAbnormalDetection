import pandas as pd
import numpy as np
import gc
import sys
import collections
from numpy.lib.stride_tricks import as_strided as strided
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


sys.path.insert(0,'/home/oem/Projects')
from Kylearn.utils.log import log_down

logfile = 'create_data_log'
logger = log_down(logfile)
gc.enable()

dev_list = ['AMP', 'ETH', 'ETH10G', 'ETHN', 'ETTP', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']
alarm_list = [None, 'Excessive Error Ratio',  # 1
              'Frequency Out Of Range',  # 2
              'GCC0 Link Failure', 'Gauge Threshold Crossing Alert Summary',  # 4
              'Link Down', 'Local Fault', 'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal',  # 9
              'OSC OSPF Adjacency Loss', 'OTU Signal Degrade',  # 11
              'Rx Power Out Of Range']  # 12
state_list = ['IS', 'n/a', 'IS-ANR']

data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_data.parquet')

scaler = StandardScaler(with_mean=False)
scaler.fit(data.iloc[:, 4:49])
le_1 = LabelEncoder()
le_1.fit(dev_list)
ohe_1 = OneHotEncoder()
ohe_1.fit(np.arange(0,len(dev_list)).reshape([-1,1]))


def slid_generate(past_days, future_days, data, state_list=None, device_list=None, alarm_list=None):
    logger.info('Past day(s): %s Future day(s): %s' % (past_days, future_days))

    devices = data[data['GROUPBYKEY'].isin(device_list)]
    logger.info('Device type: %s' % device_list)
    devices = devices[devices['ALARM'].isin(alarm_list)]  # filter out devices with alarm out of the list
    logger.info('ALARM: %s' % alarm_list)
    dev_count = devices['ID'].value_counts()
    dev_count = dev_count.drop(dev_count[dev_count < past_days + future_days].index).index.tolist()
    devices = devices[devices['ID'].isin(dev_count)]  # filter out devices that has less data than the time window
    devices = devices.drop_duplicates()
    devices = devices.set_index(['ID', 'TIME']).sort_index().reset_index().fillna(0)
    print(devices.shape)
    # drop the devices that has all zero PM values
    devices['sum'] = devices.iloc[:, 4:49].sum(axis = 1)
    devices = devices[devices['sum'] != 0]
    devices = devices.drop(['sum'], axis = 1)
    print(devices.shape)
    logger.info('DEVICE COUNT')
    logger.info('\n' + str(devices['GROUPBYKEY'].value_counts()))  # print device count
    logger.info('ALARM COUNT')
    logger.info('\n' + str(devices['ALARM'].value_counts()))  # print device count

    devices['ALARM'] = devices['ALARM'].map(lambda x: 0 if x == 0 else 1)  # mask all alarms 1

    logger.info('SAMPLE COUNT')
    logger.info('\n' + str(devices['ALARM'].value_counts()))

    def get_sliding_windows(dataFrame, windowSize, returnAs2D=False):
        stride0, stride1 = dataFrame.values.strides
        rows, columns = dataFrame.shape
        output = strided(dataFrame, shape=(rows - windowSize + 1, windowSize, columns),
                         strides=(stride0, stride0, stride1))
        if returnAs2D == 1:
            return output.reshape(dataFrame.shape[0] - windowSize + 1, -1)
        else:
            return output

    def mask_list(i):
        # If m+n data do not have the same device type, return False
        if (len(set(i[0:past_days + future_days, 3])) != 1):
            return False
        # If m data are not all in service, return False
        elif (~np.isin(i[0:past_days, 2], state_list).all()):
            return False
        # If any of m data is an anomaly instance, return False
        elif (i[0:past_days, 49].any()):
            return False
        # If i doesn't mach any of above, return True
        else:
            return True

    def label_data(i):
        # If any of the n data is an anomaly instance, return 1
        if (i[past_days:past_days + future_days, 49].any()):
            return 1
        else:
            return 0

    devices = get_sliding_windows(devices[:-1], past_days + future_days)
    mask = [mask_list(i) for i in devices]

    valid_data = devices[mask]
    label = [label_data(i) for i in valid_data]
    assert valid_data.shape[0] == len(label)
    return valid_data, label


m = 5
n = 2

X, Y = slid_generate(past_days=m, future_days=n, data=data,
                     state_list=state_list, device_list=dev_list, alarm_list=alarm_list)

devices = X[:,0,3]
devices = le_1.transform(devices)
devices = devices.reshape([-1, 1])
devices = ohe_1.transform(devices).toarray()

features = X[:,0:m,4:49]
labels = np.array(Y).reshape(-1,1)

np.save('data/m%s_n%s_attn_features.npy'%(m, n), features)
np.save('data/m%s_n%s_attn_devices.npy'%(m, n), devices)
np.save('data/m%s_n%s_attn_labels.npy'%(m, n), labels)

