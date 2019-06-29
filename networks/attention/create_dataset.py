import pandas as pd
pd.set_option('display.max_columns', 50)
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split



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
data.iloc[:, 1:46] = min_max_scaler.transform(data.iloc[:, 1:46])

device = data['GROUPBYKEY']
PMs = data.iloc[:, 1:46]
alarm = data['ALARM']

# split
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=2)
