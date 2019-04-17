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
# output = Dense(2,activation='softmax')(layer2)
#
# model = Model(input = input, output = output)
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# '''load data'''
#
# train_x = np.load('../data/%s/train_x.npy' % file_path)
# train_y_bi = np.load('../data/%s/train_y_bi.npy' % file_path)
#
#
# from sklearn.preprocessing import OneHotEncoder
#
# ohencoder = OneHotEncoder()
# train_y_bi = ohencoder.fit_transform(train_y_bi.reshape([-1, 1]))
#
# '''train'''
# from keras.callbacks import EarlyStopping
#
#
# monitor = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
# model.fit(train_x,train_y_bi,batch_size=50,epochs=1000,validation_split=0.1,callbacks=[monitor])
# model.save('../models/%s/fnnmodel'%file_path)
# del model
# del train_x,train_y_bi

#----------------------------------------------------------------------

### ASSESSMENT

'''load testset and model'''

x_test = np.load('../data/%s/test_x.npy' % file_path)
y_test_bi = np.load('../data/%s/test_y_bi.npy' % file_path)
y_test_bi = y_test_bi.reshape([-1, 1])
model = load_model('../models/%s/fnnmodel' % file_path)
'''predict'''

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).reshape([-1, 1])

'''Assessment'''

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_true=y_test_bi, y_pred=y_pred))
cm = confusion_matrix(y_true=y_test_bi, y_pred=y_pred)
acc = accuracy_score(y_true=y_test_bi, y_pred=y_pred)

'''Visualization'''

from toolPackage.draw_cm import cm_analysis

cm_analysis(cm, ['normal', 'malfunction'], precision=False)
