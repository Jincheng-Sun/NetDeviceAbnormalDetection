import pandas as pd
import numpy as np
import gc
import sys
import collections
from numpy.lib.stride_tricks import as_strided as strided
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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
data_eu = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_data.parquet')
data_eu = data_eu.drop_duplicates()
data_tk = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Tokyo_Network_Data_1Day.parquet')
data_tk = data_tk.drop_duplicates()
def unify_to(data_obj, data_sub):
    # data_sub will be convert to format of data_obj
    # label dictionary
    rep_list = {'CV-E': 'E-CV',
                'CV-PCS':'PCS-CV',
                'INFRAMESERR-E_INFRAMES-E_/': 'E-INFRAMESERR_E-INFRAMES_/',
                'ES-PCS': 'PCS-ES',
                'UAS-PCS': 'PCS-UAS',
                'SPANLOSSMAX-OCH_SPANLOSSMIN-OCH_-': 'OCH-SPANLOSSMAX_OCH-SPANLOSSMIN_-',
                'UAS-E': 'E-UAS',
                'ES-E': 'S-ES',
                'OUTFRAMESERR-E_OUTFRAMES-E_/': 'E-OUTFRAMESERR_E-OUTFRAMES_/',
                'SPANLOSSAVG-OCH': 'OCH-SPANLOSSAVG'
                }
    # rename the columns
    data_sub = data_sub.rename(columns=rep_list)

    '''Unify the columns'''
    listA = data_sub.columns.tolist()
    listB = data_obj.columns.tolist()
    diff = list(set(listB).difference(set(listA)))
    diff_columns = pd.DataFrame(columns=diff)
    data_sub = pd.concat([data_sub, diff_columns], axis=1)
    data_sub = data_sub[data_obj.columns]
    return data_obj, data_sub

# scaler and encoders
scaler = StandardScaler(with_mean=False)
scaler.fit(data_eu.iloc[:, 4:49])
le_1 = LabelEncoder()
le_1.fit(dev_list)
ohe_1 = OneHotEncoder()
ohe_1.fit(np.arange(0,len(dev_list)).reshape([-1,1]))

def slid_generate(past_days, future_days, data, state_list=None, device_list=None, alarm_list=None):
    logger.info('Past day(s): %s Future day(s): %s' % (past_days, future_days))
    #keep valid data
    devices = data[data['GROUPBYKEY'].isin(device_list)]
    logger.info('Device type: %s' % device_list)
    devices = devices[devices['ALARM'].isin(alarm_list)]  # filter out devices with alarm out of the list
    logger.info('ALARM: %s' % alarm_list)

    devices = devices.set_index(['ID', 'TIME']).sort_index().reset_index().fillna(0)


    # drop the devices that has all zero PM values
    devices['sum'] = devices.iloc[:, 4:49].sum(axis = 1)
    devices = devices[devices['sum'] != 0]
    devices = devices.drop(['sum'], axis = 1)


    dev_count = devices['ID'].value_counts()
    dev_count = dev_count.drop(dev_count[dev_count < past_days + future_days].index).index.tolist()
    devices = devices[devices['ID'].isin(dev_count)]  # filter out devices that has less data than the time window

    devices.iloc[:,4:49] = scaler.transform(devices.iloc[:,4:49].values)
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


    global a, b, c
    a, b, c = 0, 0, 0
    def mask_list(i):
        global a,b,c
        # If m+n data do not have the same device type, return False
        if (len(set(i[0:past_days + future_days, 3])) != 1):
            a+=1
            return False
        # If m data are not all in service, return False
        elif (~np.isin(i[0:past_days, 2], state_list).all()):
            b+=1
            return False
        # If any of m data is an anomaly instance, return False
        elif (i[0:past_days, 49].any()):
            c+=1
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
    print(a,b,c)
    print(collections.Counter(mask))

    valid_data = devices[mask]
    label = [label_data(i) for i in valid_data]
    assert valid_data.shape[0] == len(label)
    print(collections.Counter(label))
    return valid_data, label

def slid_generate_un(past_days, future_days, data, state_list=None, device_list=None):
    logger.info('Past day(s): %s Future day(s): %s' % (past_days, future_days))

    devices = data[data['GROUPBYKEY'].isin(device_list)]
    logger.info('Device type: %s' % device_list)
    devices = devices.set_index(['ID', 'TIME']).sort_index().reset_index().fillna(0)
    # drop the devices that has all zero PM values
    devices['sum'] = devices.iloc[:, 4:49].sum(axis = 1)
    devices = devices[devices['sum'] != 0]
    devices = devices.drop(['sum'], axis = 1)
    dev_count = devices['ID'].value_counts()
    dev_count = dev_count.drop(dev_count[dev_count < past_days + future_days].index).index.tolist()
    devices = devices[devices['ID'].isin(dev_count)]  # filter out devices that has less data than the time window

    # Try one with out scaler
    devices.iloc[:,4:49] = scaler.transform(devices.iloc[:,4:49].values)
    print(devices.shape)
    logger.info('DEVICE COUNT')
    logger.info('\n' + str(devices['GROUPBYKEY'].value_counts()))  # print device count


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
        else:
            return True

    devices = get_sliding_windows(devices[:-1], past_days + future_days)
    mask = [mask_list(i) for i in devices]

    valid_data = devices[mask]
    print(valid_data.shape)

    return valid_data

data_eu, data_tk = unify_to(data_eu, data_tk)

# Use past m days of data to predict n days ahead
m = 3
n = 2

# X has a shape of [?, m+n, 50], Y has a shape of [?, 1].
# If modified to generate data for classification, Y should have a shape of [?, 12](one-hotted)
X, Y = slid_generate(past_days=m, future_days=n, data=data_eu,
                     state_list=state_list, device_list=dev_list, alarm_list=alarm_list)

devices = X[:,0,3]
# turn string device type to one-hot representation
devices = le_1.transform(devices)
devices_train, devices_test = train_test_split(devices, test_size=0.2, random_state=23)


# extract features from X, features should have a shape of [?, m, 45]
features = X[:,0:m,4:49]
features = np.expand_dims(features, axis=-1)
features_train, features_test = train_test_split(features, test_size=0.2, random_state=23)
# For classification it should be reshape to [-1,12]
labels = np.array(Y)
labels_train, labels_test = train_test_split(labels, test_size=0.2, random_state=23)

# save file
np.save('data/m%s_n%s_X_train.npy'%(m, n), features_train)
np.save('data/m%s_n%s_dev_train.npy'%(m, n), devices_train)
np.save('data/m%s_n%s_y_train.npy'%(m, n), labels_train)

np.save('data/m%s_n%s_X_test.npy'%(m, n), features_test)
np.save('data/m%s_n%s_dev_test.npy'%(m, n), devices_test)
np.save('data/m%s_n%s_y_test.npy'%(m, n), labels_test)


X_un = slid_generate_un(past_days=m, future_days=n, data=data_tk,
                     state_list=state_list, device_list=dev_list)

devices_un = X_un[:, 0, 3]
# turn string device type to one-hot representation
devices_un = le_1.transform(devices_un)

# extract features from X, features should have a shape of [?, m, 45]
features_un = X_un[:, 0:m, 4:49]
features_un = np.expand_dims(features_un, axis=-1)

# For classification it should be reshape to [-1,12]

# save file
np.save('data/m%s_n%s_X_un_train.npy' % (m, n), features_un)
np.save('data/m%s_n%s_dev_un_train.npy' % (m, n), devices_un)

