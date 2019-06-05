import pandas as pd
import numpy as np

pm = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_network_data.parquet')
alarm = pd.read_parquet('/home/oem/Projects/NetDeviceAbnormalDetection/data/Europe_network_labels.parquet')
alarm = alarm.loc[~alarm['description'].str.contains('Demo for Josh')]

alarm['TIME'] = pd.to_datetime(alarm['time'], infer_datetime_format=True).dt.floor('1D')
alarm = alarm.drop(['description','time','timestamp','extraAttributes'], axis=1).rename({'category':'ALARM'}, axis=1)
alarm = alarm.drop_duplicates()

# optmon = data[data['GROUPBYKEY']=='OPTMON']
# agragation = optmon.set_index(['ID','TIME']).sort_index().reset_index()

