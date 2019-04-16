import numpy as np
import pandas as pd
'''load dataset'''
x_mal = pd.DataFrame(np.load('../data/mal_x.npy'))
x_normal = pd.DataFrame(np.load('../data/nor_x.npy'))
y_mal = pd.DataFrame(np.load('../data/mal_y.npy'))
y_normal = pd.DataFrame(np.load('../data/nor_y.npy'))

'''split dataset'''
from sklearn.model_selection import train_test_split

X_nor,x_nor,Y_nor,y_nor = train_test_split(x_normal, y_normal, test_size=y_mal.shape[0], random_state=42)

'''concatenate'''
x_mal = pd.concat([x_nor, x_mal], axis=0)
y_mal = pd.concat([y_nor, y_mal], axis=0)
# del X_nor,x_nor,Y_nor,y_nor

'''split the data'''
X,x,Y,y = train_test_split(x_mal, y_mal, test_size=0.2, random_state=42)
assert (Y[0].value_counts().shape == y[0].value_counts().shape)

'''encode the data'''
from keras.models import load_model

encoder = load_model('../models/encoder')
X = encoder.predict(X)
x = encoder.predict(x)

np.save('../data/train_x.npy',X)
np.save('../data/test_x.npy',x)

np.save('../data/train_y_all.npy',Y)
np.save('../data/test_y_all.npy',y)

train_bi = Y[0].map(lambda x: 1 if x!=0 else 0)
test_bi = y[0].map(lambda x: 1 if x!=0 else 0)

np.save('../data/train_y_bi',train_bi)
np.save('../data/test_y_bi',test_bi)

