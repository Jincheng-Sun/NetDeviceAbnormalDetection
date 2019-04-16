from keras import Model
from keras.layers import Input, Dense
from keras import Model
from sklearn.externals import joblib
import numpy as np
'''global variance'''
file_path = 'aftconcat'

# '''build model'''
#
# input_classification = Input(shape=(31,))
# input_binary = Input(shape=(31,))
#
#
# def share(input):
#     layer1 = Dense(64, activation='relu')(input)
#     layer2 = Dense(128, activation='relu')(layer1)
#     return layer2
#
#
# mid_classification = share(input_classification)
# mid_binary = share(input_binary)
#
# output_classification = Dense(16, activation='softmax')(mid_classification)
# output_binary = Dense(2, activation='softmax')(mid_binary)
#
# model = Model(input=[input_classification, input_binary], output=[output_classification, output_binary])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
#
# '''load data'''
# file_path = 'aftconcat'
# train_x = np.load('../data/%s/train_x.npy' % file_path)
# train_y_all = np.load('../data/%s/train_y_all.npy' % file_path)
# train_y_bi = np.load('../data/%s/train_y_bi.npy' % file_path)
#
# labelencoder = joblib.load('../models/labelencoder')
# ohencoder = joblib.load('../models/OneHotEncoder')
#
# train_y_all = labelencoder.transform(train_y_all).reshape([-1, 1])
# train_y_all = ohencoder.fit_transform(train_y_all)
#
# del labelencoder, ohencoder
#
# from sklearn.preprocessing import OneHotEncoder
#
# ohencoder = OneHotEncoder()
# train_y_bi = ohencoder.fit_transform(train_y_bi.reshape([-1, 1]))
#
# '''split dataset'''
# from sklearn.model_selection import train_test_split
#
# x_train, _, y_train, _ = train_test_split(train_x, train_y_all, test_size=0, random_state=42)
#
# x_train2, _, y_train2, _ = train_test_split(train_x, train_y_bi, test_size=0, random_state=43)
# del train_x, train_y_all, train_y_bi
#
# #-------------------------------------------------------------------
# '''train'''
# from keras.callbacks import EarlyStopping
#
# monitor = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# model.fit(x=[x_train, x_train2], y=[y_train, y_train2], batch_size=50, epochs=1000, callbacks=[monitor],validation_split=0.1)
# model.save('../models/%s/oneclassmodel'%file_path)
#
# # -------------------------------------------------------------------
import pandas as pd
from keras.models import load_model


'''load testset and encoders'''
x_test = np.load('../data/%s/test_x.npy' % file_path)
y_test_all = np.load('../data/%s/test_y_all.npy' % file_path)
y_test_bi = np.load('../data/%s/test_y_bi.npy' % file_path)
labelencoder = joblib.load('../models/labelencoder')
model = load_model('../models/%s/oneclassmodel' % file_path)

y_test_bi = y_test_bi.reshape([-1, 1])
y_test_all = labelencoder.transform(y_test_all).reshape([-1, 1])
label = labelencoder.classes_.tolist()

'''predict'''
y_pred_all, y_pred_bi = model.predict([x_test, x_test])
y_pred_all = np.argmax(y_pred_all, axis=1).reshape([-1, 1])
y_pred_bi = np.argmax(y_pred_bi, axis=1).reshape([-1, 1])
'''Assessment'''
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(classification_report(y_true=y_test_all, y_pred=y_pred_all))
cm = confusion_matrix(y_true=y_test_all, y_pred=y_pred_all)
acc = accuracy_score(y_true=y_test_all, y_pred=y_pred_all)
'''Visualization'''
import seaborn as sns
import matplotlib.pyplot as plt


from toolPackage.draw_cm import cm_analysis

cm_analysis(cm, label, precision=False)

'''for binary'''
print(classification_report(y_true=y_test_bi, y_pred=y_pred_bi))
cm2 = confusion_matrix(y_true=y_test_bi, y_pred=y_pred_bi)
acc2 = accuracy_score(y_true=y_test_bi, y_pred=y_pred_bi)

cm_analysis(cm2, ['normal', 'malfunction'], x_rotation=0, font_size=0.5, precision=False)

