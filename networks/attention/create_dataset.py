import pandas as pd
pd.set_option('display.max_columns', 50)
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

min_max_scaler = joblib.load('/home/oem/Projects/NetDeviceAbnormalDetection/models/Minmax_scaler')
# autoencoder = load_model('encoders/encoder_1layer_75dims')
# label_encoder = joblib.load('encoders/label_encoder')
# onehot_encoder = joblib.load('encoders/onehot_encoder')

data = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_Network_Data_May13.parquet')
dev_list = ['CHMON', 'STM64', 'OC192', 'STTP', 'STM4', 'STM16', 'NMCMON', 'OC48', 'OC12', 'OC3', 'FLEX', 'RAMAN']
alarm_list = ['Excessive Error Ratio', 'Frequency Out Of Range', 'GCC0 Link Failure',
              'Gauge Threshold Crossing Alert Summary', 'Link Down', 'Local Fault',
              'Loss Of Clock', 'Loss Of Frame', 'Loss Of Signal', 'OSC OSPF Adjacency Loss',
              'OTU Signal Degrade', 'Rx Power Out Of Range ']
data = data.drop(data[data['GROUPBYKEY'].isin(dev_list)].index)
data = data[data['ALARM'].isin(alarm_list)]
data = data.fillna(0)

data = data.drop(['ID','TIME','LABEL'], axis=1)
# data.iloc[:, 1:46] = min_max_scaler.transform(data.iloc[:, 1:46])
# split
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=2)
train = data[train_idx]
test = data[test_idx]

def process(data):
    PMs = data.iloc[:, 1:46]
    # Min-max scaling PM values
    PMs = min_max_scaler.transform(PMs)

    # device onehot-encoding
    device = data['GROUPBYKEY']
    le_1 = LabelEncoder()
    ohe = OneHotEncoder()
    device = le_1.fit_transform(device)
    device = ohe.fit(device)

    # alarm label-to-number
    alarm = data['ALARM']
    le_2 = LabelEncoder()
    alarm = le_2.fit_transform(alarm)
    return PMs, device, alarm

X1, dev1, y1 = process(train)
np.save('attn_path',X1)
np.save('attn_path',dev1)
np.save('attn_path',y1)

X2, dev2, y2 = process(test)
np.save('attn_path',X2)
np.save('attn_path',dev2)
np.save('attn_path',y2)