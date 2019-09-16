import pandas as pd
import numpy as np
import gc
import sys
import collections
from numpy.lib.stride_tricks import as_strided as strided
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

pd.options.display.max_columns = 300

sys.path.insert(0,'/home/oem/Projects')
from Kylearn.utils.log import log_down

logfile = 'create_data_log'
logger = log_down(logfile)
gc.enable()

# device list, alarm_list and state_list for filtering the dataset
dev_list = ['AMP', 'ETH', 'ETH10G', 'ETHN', 'ETTP', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']
alarm_list = [None, 'Excessive Error Ratio',  # 1
              'Frequency Out Of Range',  # 2
              'GCC0 Link Failure', 'Gauge Threshold Crossing Alert Summary',  # 4
              'Link Down', 'Local Fault', 'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal',  # 9
              'OSC OSPF Adjacency Loss', 'OTU Signal Degrade',  # 11
              'Rx Power Out Of Range']  # 12
state_list = ['IS', 'n/a', 'IS-ANR']

# load europe dataset
# dataset should be Europe_Network_data from 2018.11 to 2019.03 that Derek shared on March 25
# do not use May13
data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/colt_europe_all_pms_Feb_Jul_2019_w_netcool_label.parquet')
data = data.drop_duplicates()
data = data.rename(columns = {  'meta_facility':'GROUPBYKEY',
                                'meta_status':'LABEL',
                                'category':'ALARM',
                              })


# scaler and encoders
scaler = StandardScaler(with_mean=False)
scaler.fit(data.iloc[:, 3:214])
le_1 = LabelEncoder()
le_1.fit(dev_list)
ohe_1 = OneHotEncoder()
ohe_1.fit(np.arange(0,len(dev_list)).reshape([-1,1]))

def slid_generate(past_days, future_days, data, state_list=None, device_list=None, alarm_list=None):
    logger.info('Past day(s): %s Future day(s): %s' % (past_days, future_days))

    devices = data[data['GROUPBYKEY'].isin(device_list)]
    logger.info('Device type: %s' % device_list)
    # devices = devices[devices['ALARM'].isin(alarm_list)]  # filter out devices with alarm out of the list
    # logger.info('ALARM: %s' % alarm_list)
    dev_count = devices['ID'].value_counts()
    dev_count = dev_count.drop(dev_count[dev_count < past_days + future_days].index).index.tolist()
    devices = devices[devices['ID'].isin(dev_count)]  # filter out devices that has less data than the time window
    devices = devices.drop_duplicates()
    devices = devices.set_index(['ID', 'TIME']).sort_index().reset_index().fillna(0)
    print(devices.shape)
    # drop the devices that has all zero PM values
    devices['sum'] = devices.iloc[:, 3:214].sum(axis = 1)
    devices = devices[devices['sum'] != 0]
    devices = devices.drop(['sum'], axis = 1)
    print(devices.shape)
    logger.info('DEVICE COUNT')
    logger.info('\n' + str(devices['GROUPBYKEY'].value_counts()))  # print device count
    logger.info('ALARM COUNT')
    logger.info('\n' + str(devices['ALARM'].value_counts()))  # print device count
    # ------------------------------------------------------------------------------
    # change this line to generate label range from 0 ~ 11 (since there 12 kinds of alarm
    devices['ALARM'] = devices['ALARM'].map(lambda x: 0 if x == 0 else 1)  # mask all alarms 1
    # ------------------------------------------------------------------------------

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
        # # If m data are not all in service, return False
        # elif (~np.isin(i[0:past_days, 2], state_list).all()):
        #     return False
        # # If any of m data is an anomaly instance, return False
        # elif (i[0:past_days, -1].any()):
        #     return False
        # If i doesn't mach any of above, return True
        else:
            return True

    def label_data(i):
        # If any of the n data is an anomaly instance, return 1
        if (i[past_days:past_days + future_days, -1].any()):
            return 1
        else:
            return 0

    devices = get_sliding_windows(devices[:-1], past_days + future_days)
    mask = [mask_list(i) for i in devices]

    valid_data = devices[mask]
    label = [label_data(i) for i in valid_data]
    assert valid_data.shape[0] == len(label)
    return valid_data, label

# Use past m days of data to predict n days ahead
m = 3
n = 0

# X has a shape of [?, m+n, 50], Y has a shape of [?, 1].
# If modified to generate data for classification, Y should have a shape of [?, 12](one-hotted)
X, _ = slid_generate(past_days=m, future_days=n, data=data,
                     state_list=state_list, device_list=dev_list, alarm_list=alarm_list)

features = X[:, :, 3:214]
labels = X[:, :, -1]

inputs = features[:, :m]
targets = features[:, -1]
targets = np.expand_dims(targets, 1)
labels = labels[:, -1]
labels = np.expand_dims(labels, -1)


# labels = np.expand_dims(labels, axis=-1)
# func = lambda x: [0, 1] if x == 1 else [1, 0]
# one_hot_labels = np.apply_along_axis(func, axis = 2, arr=labels)


# devices = X[:,0,3]
# # turn string device type to one-hot representation
# devices = le_1.transform(devices)
# devices = devices.reshape([-1, 1])
# devices = ohe_1.transform(devices).toarray()
#
# # extract features from X, features should have a shape of [?, m, 45]
# features = X[:,0:m,3:214]
# # For classification it should be reshape to [-1,12]
# labels = np.array(Y).reshape(-1,1)
#
# save file
np.save('data/all_pm/m%s_features.npy'%m, inputs)
np.save('data/all_pm/n%s_targets.npy'%n, targets)
np.save('data/all_pm/n%s_labels.npy'%n, labels)

