import keras
from keras.layers import Dense, Input
from keras.models import load_model
from keras import Model
import numpy as np
from sklearn.externals import joblib

'''global variance'''
file_path = 'origindata'
# '''build model'''
# input = Input(shape=(56,))
# layer1 = Dense(64,activation='relu')(input)
# layer2 = Dense(128,activation='relu')(layer1)
# output = Dense(16,activation='softmax')(layer2)
#
# model = Model(input = input, output = output)
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# '''load data'''
#
# train_x = np.load('../data/%s/train_x.npy' % file_path)
# train_y_all = np.load('../data/%s/train_y_all.npy' % file_path)
#
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
# '''turn label into one-hot form'''
# labelencoder = LabelEncoder()
# train_y_all = labelencoder.fit_transform(train_y_all)
# train_y_all = train_y_all.reshape([-1, 1])
# ohencoder = OneHotEncoder()
# train_y_all = ohencoder.fit_transform(train_y_all)
#
# '''save encoders'''
#
# joblib.dump(labelencoder,'../models/labelencoder')
# joblib.dump(ohencoder,'../models/OneHotEncoder')
# del labelencoder,ohencoder
#
# '''split train-valid set'''
# from sklearn.model_selection import train_test_split
# x_train,x_val,y_train,y_val = train_test_split(train_x, train_y_all, test_size=0.1, random_state=42)
# del train_x,train_y_all
#
#
# '''train'''
# from keras.callbacks import EarlyStopping
#
#
# monitor = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
# model.fit(x_train,y_train,batch_size=50,epochs=1000,validation_data=(x_val,y_val),callbacks=[monitor])
# model.save('../models/%s/fnnmodel'%file_path)
# del model
# del x_train,x_val,y_train,y_val
# ----------------------------------------------------------------------

### ASSESSMENT

'''load testset and encoders'''

x_test = np.load('../data/%s/test_x.npy' % file_path)
y_test = np.load('../data/%s/test_y_all.npy' % file_path)
labelencoder = joblib.load('../models/labelencoder')
ohencoder = joblib.load('../models/OneHotEncoder')
model = load_model('../models/%s/fnnmodel' % file_path)
y_test = labelencoder.transform(y_test).reshape([-1, 1])
label = labelencoder.classes_.tolist()

'''predict'''

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).reshape([-1, 1])

'''Assessment'''

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_true=y_test, y_pred=y_pred))
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)

'''Visualization'''

from toolPackage.draw_cm import cm_analysis

cm_analysis(cm, label, precision=False)
