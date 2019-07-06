import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from keras.models import load_model

# Devices we focus on
dev_list = ['AMP', 'ETH', 'ETH10G', 'ETHN', 'ETTP', 'OPTMON', 'OSC', 'OTM', 'OTM2', 'OTUTTP', 'PTP']
# Alarms we focus on, note that `Laser Off Far End Failure Triggered` and 'Remote Fault' will be excluded
# as discussed with David
alarm_list = ['Excessive Error Ratio',
              'Frequency Out Of Range',
              'GCC0 Link Failure', 'Gauge Threshold Crossing Alert Summary',
              'Laser Off Far End Failure Triggered', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal',
              'OSC OSPF Adjacency Loss', 'OTU Signal Degrade',
              'Remote Fault', 'Rx Power Out Of Range']
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
    normal_labeled, normal_unlabeled = train_test_split(
        normal, train_size=anomaly.shape[0], random_state=42)

    return anomaly, normal_labeled, normal_unlabeled


def label_data(anomaly_data, is_classification=True):
    # input data is the anomaly data partition
    # if it's for classification problem, turn the alarms to numerical labels
    # else label them 1
    if is_classification:
        le = LabelEncoder()
        le.fit(alarm_list)
        anomaly_data['ALARM'] = le.transform(anomaly_data['ALARM']).reshape([-1, 1])
    else:
        anomaly_data['ALARM'] = 1
    return anomaly_data

def mask_data(unlabeled_data):
    # input data is the unlabeled data
    # mask the alarm as -1
    unlabeled_data['ALARM'] = -1
    return unlabeled_data

# preprocessing flow in evaluation.ipynb
min_max_scaler = joblib.load('encoders/Minmax_scaler')
autoencoder = load_model('encoders/encoder_1layer_75dims')
label_encoder = joblib.load('encoders/label_encoder')
onehot_encoder = joblib.load('encoders/onehot_encoder')

def preprocessing(data):
    # drop useless columns
    data = data.drop(['ID', 'TIME', 'LABEL'], axis=1)
    # scaling the data into interval [0,1].
    # Using min_max scaler change the center of input distribution,
    # consider using StandardScaler with with_mean = False
    data.iloc[:, 1:46] = min_max_scaler.transform(data.iloc[:, 1:46])
    PMs = autoencoder.predict(data.iloc[:, 1:46])
    GBK = label_encoder.transform(np.reshape(data['GROUPBYKEY'].tolist(), [-1, 1]))
    GBK = np.reshape(GBK, [-1, 1])
    GBK = onehot_encoder.transform(GBK)
    GBK = pd.DataFrame(GBK.toarray())
    X = np.concatenate([GBK, PMs], axis=1)
    X = X.reshape(X.shape + [1])
    y = data['ALARM']
    return X, y



# -------------------------------------------------------------------------------------------
# 1. For present tense classification, with Mean-Teacher frame work,
#    we need both labeled data (from Europe dataset) and unlabeled data (from Tokyo dataset).
# -------------------------------------------------------------------------------------------

# load Tokyo and Europe dataset
raw_data_tk = pd.read_parquet('data/Tokyo_Network_Data_1Day.parquet')
raw_data_eu = pd.read_parquet('data/Europe_Network_data.parquet')

raw_data_eu, raw_data_tk = unify_to(raw_data_eu, raw_data_tk)

raw_data_eu = keep_valid_data(raw_data_eu)
# for classification, we only need the anomaly data and the unlabeled data
anomaly_eu, _, normal_unlabeled_eu = split_and_drop(raw_data_eu)
anomaly_eu = label_data(anomaly_eu, is_classification=True)
normal_unlabeled_eu = mask_data(normal_unlabeled_eu)
# take all the Tokyo data as unlabeled data
raw_data_tk = keep_valid_data(raw_data_tk)
unlabeled_tk = mask_data(raw_data_tk)
# create dataset
# split anomaly data, keep the second part as testset since we don't need unlabeled data for testing
trainset_1, testset= train_test_split(anomaly_eu, test_size=0.2, random_state=22)
# combine the two unlabeled data partition
trainset_2 = pd.concat([normal_unlabeled_eu, unlabeled_tk], axis=0)
# combine and shuffle the dataset
trainset = pd.concat([trainset_1, trainset_2], axis=0).sample(frac=1).reset_index()
print(trainset['ALARM'].value_counts())
# Apply the preprocessing flow in evaluation.ipynb after
X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)
# save data in npy format
np.save('X_train_path.npy', X_train)


# -------------------------------------------------------------------------------------------
# 2. For present tense anomaly detection
# -------------------------------------------------------------------------------------------

# load Tokyo and Europe dataset
raw_data_tk = pd.read_parquet('data/Tokyo_Network_Data_1Day.parquet')
raw_data_eu = pd.read_parquet('data/Europe_Network_data.parquet')

raw_data_eu, raw_data_tk = unify_to(raw_data_eu, raw_data_tk)
raw_data_eu = keep_valid_data(raw_data_eu)
# for anomaly detection, we need the anomaly and normal data as well as the unlabeled data
anomaly_eu, normal_labeled, normal_unlabeled_eu = split_and_drop(raw_data_eu)
anomaly_eu = label_data(anomaly_eu, is_classification=False)
normal_unlabeled_eu = mask_data(normal_unlabeled_eu)
# take all the Tokyo data as unlabeled data
raw_data_tk = keep_valid_data(raw_data_tk)
unlabeled_tk = mask_data(raw_data_tk)
# create dataset
# combine anomaly and equal amount of normal data, and split
trainset_1, testset= train_test_split(
    pd.concat([anomaly_eu, normal_labeled], axis=0), test_size=0.2, random_state=22)
# combine the two unlabeled data partition
trainset_2 = pd.concat([normal_unlabeled_eu, unlabeled_tk], axis=0)
# combine and shuffle the dataset
trainset = pd.concat([trainset_1, trainset_2], axis=0).sample(frac=1).reset_index()
print(trainset['ALARM'].value_counts())
# Apply the preprocessing flow in evaluation.ipynb after
X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)
# save data in npy format
np.save('X_train_path.npy', X_train)