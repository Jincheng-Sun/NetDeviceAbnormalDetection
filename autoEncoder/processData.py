import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
pd.set_option('display.max_columns', 50)
data = pd.read_parquet('../data/Europe_Network_data.parquet')
data = data.drop(['ID', 'TIME', 'LABEL'], axis=1)
scaler = pre.MinMaxScaler()
data.iloc[:,1:46] = scaler.fit_transform(data.iloc[:,1:46])
data = data.fillna(0)
# new = pd.concat([pd.get_dummies(data['GROUPBYKEY']), data], axis=1)
new = data.drop(['GROUPBYKEY'], axis=1)
new.head(5)
np.save('../data/scaler',scaler)

df1 =new.loc[new["ALARM"] != 0]
df2 = new.drop(df1.index)
()
np.save('../data/x_encoder',new.drop(['ALARM'], axis=1))

mal_x = df1.drop(['ALARM'], axis=1)
mal_y = df1['ALARM']
nor_x = df2.drop(['ALARM'], axis=1)
nor_y = df2['ALARM']

np.save('../data/mal_x',mal_x)
np.save('../data/mal_y',mal_y)
np.save('../data/nor_x',nor_x)
np.save('../data/nor_y',nor_y)