import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.externals import joblib

pd.set_option("display.max_columns", 228)

all_pms = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/colt_europe_all_pms_Feb_Jul_2019_w_netcool_label.parquet')
all_pms = all_pms.drop_duplicates()
# all_pms = all_pms.dropna(axis=0, how='all')
# all_pms.shape
# Out[3]: (3385060, 228)


# First step, drop those data with no alarm and is one of the oos
all_pms = all_pms.fillna(0)

status_mask = all_pms.iloc[:, -5].isin(['OOS-AU', 'OOS-AUMA'])

targets_mask = all_pms.iloc[:, -1] == 0

mask = status_mask & targets_mask

all_pms = all_pms[~mask]

del status_mask, targets_mask

status = all_pms.iloc[:, -5]

targets = all_pms.iloc[:, -1]

targets_binary = targets.copy()

targets_binary = targets_binary != 0

targets_binary = targets_binary.astype(int)

targets = targets.replace(0, 'normal')



features = all_pms.iloc[:, 3:-14]

device_type = all_pms.iloc[:, 2]

del all_pms

# # Second, rescale features, encode device type and targets
#
# feature_scaler = StandardScaler()
# dev_label_en = LabelEncoder()
# dev_oh_en = OneHotEncoder()
# target_label_en = LabelEncoder()
# target_oh_en = OneHotEncoder()
#
# features = feature_scaler.fit_transform(features)
# device_type = dev_label_en.fit_transform(device_type)
# device_type = dev_oh_en.fit_transform(device_type.reshape(-1,1))
# device_type = device_type.toarray()
# targets = target_label_en.fit_transform(targets)
# targets = target_oh_en.fit_transform(targets.reshape(-1,1))
# targets = targets.toarray()
#
# joblib.dump(feature_scaler,'models/feature_scaler')
# joblib.dump(dev_label_en,'models/dev_label_en')
# joblib.dump(dev_oh_en,'models/dev_oh_en')
# joblib.dump(target_label_en,'models/target_label_en')
# joblib.dump(target_oh_en,'models/target_oh_en')
#
#
# # last, split and save data
#
# feature_train, feature_test, device_train, device_test, target_train, target_test, target_binary_train, \
# target_binary_test = train_test_split(features, device_type, targets, targets_binary, test_size=0.2, random_state=25)
#
# np.save('data/feature_train',feature_train)
# np.save('data/feature_test',feature_test)
# np.save('data/device_train',device_train)
# np.save('data/device_test',device_test)
# np.save('data/target_train',target_train)
# np.save('data/target_test',target_test)
# np.save('data/target_binary_train',target_binary_train)
# np.save('data/target_binary_test',target_binary_test)