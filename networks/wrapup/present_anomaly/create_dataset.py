import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


# load Tokyo and Europe dataset
raw_data_tk = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Tokyo_Network_Data_1Day.parquet')
# raw_data_tk = raw_data_tk.drop_duplicates()
raw_data_eu = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_data.parquet')
# raw_data_eu = raw_data_eu.drop_duplicates()
# Devices we focus on
dev_list = ['AMP', 'ETH', 'ETH10G', 'ETHN', 'ETTP', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']
# Alarms we focus on, note that `Laser Off Far End Failure Triggered` and 'Remote Fault' will be excluded
# as discussed with David
alarm_list = ['Excessive Error Ratio',  # 1
              'Frequency Out Of Range',  # 2
              'GCC0 Link Failure', 'Gauge Threshold Crossing Alert Summary',  # 4
              'Link Down', 'Local Fault', 'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal',  # 9
              'OSC OSPF Adjacency Loss', 'OTU Signal Degrade',  # 11
              'Rx Power Out Of Range']  # 12
state_list = ['IS', 'n/a', 'IS-ANR']

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

def keep_valid_data(raw_data):
    # keep data of certain devices
    raw_data = raw_data[raw_data['GROUPBYKEY'].isin(dev_list)]
    return raw_data


def split_and_drop(raw_data):
    # keep data of certain alarms and normal data
    raw_data = raw_data[raw_data['ALARM'].isin(alarm_list+[None])]
    raw_data = raw_data.fillna(0)
    # divide dataset into normal and anomaly group
    normal = raw_data[raw_data['ALARM'] == 0]
    anomaly = raw_data[raw_data['ALARM'] != 0]
    print(normal.shape)
    print(anomaly.shape)
    # normal data should be 'in service'
    normal = normal[normal['LABEL'].isin(state_list)]
    # random sample same amount with anomaly data of normal data
    # as labeled data, and the rest as unlabeled data.
    # normal_labeled, normal_unlabeled = train_test_split(
    #     normal, train_size=anomaly.shape[0], random_state=42)

    # return anomaly, normal_labeled, normal_unlabeled
    return anomaly, normal

scaler = StandardScaler(with_mean=False)
scaler.fit(raw_data_eu.iloc[:, 4:49])
le_1 = LabelEncoder()
le_1.fit(dev_list)
ohe_1 = OneHotEncoder()
ohe_1.fit(np.arange(0, len(dev_list)).reshape([-1, 1]))
le_2 = LabelEncoder()
le_2.fit(alarm_list)

def label_data(anomaly_data, is_classification=True):
    # input data is the anomaly data partition
    # if it's for classification problem, turn the alarms to numerical labels
    # else label them 1
    if is_classification:
        anomaly_data['ALARM'] = le_2.transform(anomaly_data['ALARM']).reshape([-1, 1])
    else:
        anomaly_data['ALARM'] = 1
    return anomaly_data

def mask_data(unlabeled_data):
    # input data is the unlabeled data
    # mask the alarm as -1
    unlabeled_data['ALARM'] = -1
    return unlabeled_data


def preprocessing(data):
    # drop useless columns
    data = data.drop(['ID', 'TIME', 'LABEL'], axis=1)
    # scaling the data into interval [0,1].
    # Using min_max scaler change the center of input distribution,
    # consider using StandardScaler with with_mean = False
    # Use scaler
    # PMs = data.iloc[:, 1:46].values
    PMs = scaler.transform(data.iloc[:, 1:46])


    GBK = le_1.transform(data['GROUPBYKEY'].tolist())
    # GBK = np.reshape(GBK, [-1, 1])
    # GBK = ohe_1.transform(GBK)
    y = data['ALARM'].values
    return PMs, GBK, y




raw_data_eu, raw_data_tk = unify_to(raw_data_eu, raw_data_tk)
raw_data_eu = keep_valid_data(raw_data_eu)

# -----------------------------------------------------------------------------------
# for a balanced train set and an imbalanced test set.
# -----------------------------------------------------------------------------------

# anomaly_eu, normal_eu = split_and_drop(raw_data_eu)
# anomaly_eu = label_data(anomaly_eu, is_classification=False)
# anomaly_eu_train, anomaly_eu_test = train_test_split(anomaly_eu, test_size=0.2, random_state=23)
# normal_eu_train, normal_eu_test = train_test_split(normal_eu, test_size=0.2, random_state=23)
# normal_eu_train, unlabeled_eu = train_test_split(normal_eu_train,
#                                                         train_size=anomaly_eu_train.shape[0], random_state=23)
# raw_data_tk = keep_valid_data(raw_data_tk)
# raw_data_tk = raw_data_tk.fillna(0)
# unlabeled_tk = mask_data(raw_data_tk)
#
# trainset = pd.concat([anomaly_eu_train,normal_eu_train], axis=0)
# testset = pd.concat([anomaly_eu_test,normal_eu_test], axis=0)
# trainset_unlabeled = pd.concat([unlabeled_eu, unlabeled_tk])

# -----------------------------------------------------------------------------------
# for an imbalanced train set and an imbalanced test set.
# -----------------------------------------------------------------------------------

anomaly_eu, normal_eu = split_and_drop(raw_data_eu)
anomaly_eu = label_data(anomaly_eu, is_classification=False)
anomaly_eu_train, anomaly_eu_test = train_test_split(anomaly_eu, test_size=0.2, random_state=23)
normal_eu_train, normal_eu_test = train_test_split(normal_eu, test_size=0.2, random_state=23)
raw_data_tk = keep_valid_data(raw_data_tk)
raw_data_tk = raw_data_tk.fillna(0)
unlabeled_tk = mask_data(raw_data_tk)

trainset = pd.concat([anomaly_eu_train,normal_eu_train], axis=0)
testset = pd.concat([anomaly_eu_test,normal_eu_test], axis=0)
trainset_unlabeled = unlabeled_tk

print(trainset['ALARM'].value_counts())
# Apply the preprocessing flow in evaluation.ipynb after
X_train, dev_train, y_train = preprocessing(trainset)
X_test, dev_test, y_test = preprocessing(testset)
X_train_un, dev_train_un, _ = preprocessing(trainset_unlabeled)
# Reshape and expand
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_train_un = np.expand_dims(X_train_un, axis=-1)
# save data in npy format
# training set
np.save('data/X_train.npy', X_train)
np.save('data/dev_train.npy', dev_train)
np.save('data/y_train.npy', y_train)
#test set
np.save('data/X_test.npy', X_test)
np.save('data/dev_test.npy', dev_test)
np.save('data/y_test.npy', y_test)
# unlabeled
np.save('data/X_un_train.npy', X_train_un)
np.save('data/dev_un_train.npy', dev_train_un)



