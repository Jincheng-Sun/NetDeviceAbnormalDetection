from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, add, Input, Dense, Flatten
from keras import Model
from keras.optimizers import RMSprop
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import load_model
import sys
sys.path.insert(0,'/home/oem/Projects/Kylearn')
from examples.ciena.ciena_pred_dataset import *
from visualization.draw_matrix import *


def bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params['conv_activation'])(layer)

    if dropout > 0:
        layer = Dropout(dropout)(layer)
    return layer


def resnet_block(layer, filters, kernels, dropout, activation,
                 cross_block=False, is_first=False, is_last=False, shrink=False):
    # -BN-Act-Conv-BN-Act-Conv--
    # ↳-----------------------↑
    strides = 1
    if shrink:
        strides = 2

    if cross_block:

        shortcut = Conv2D(filters=filters,
                          kernel_size=strides,
                          kernel_initializer='random_uniform',
                          strides=strides,
                          padding='same')(layer)
    else:
        shortcut = layer

    if not is_first:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   strides=strides,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   strides=1,
                   padding='same')(layer)
    layer = add([shortcut, layer])

    if is_last:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    return layer


'''dataset'''

device_type = 'OPTMON'
dataset = pred_Dataset_2(x_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_pms_3_partial_may.npy'%device_type,
                    y_path= '/home/oem/Projects/NetDeviceAbnormalDetection/data/perdevice/%s_alarms_2days_may.npy'%device_type)


'''create model'''
input = Input(shape=(3, 5, 1))
layer = Conv2D(filters=32,
               kernel_size=3,
               kernel_initializer='random_uniform',
               strides=1,
               padding='same')(input)
layer = resnet_block(layer=layer, filters=32, kernels=3, dropout=0, activation='relu')

layer = resnet_block(layer, 64, 3, 0, 'relu', cross_block=True, shrink=True)
layer = resnet_block(layer, 64, 3, 0, 'relu')
#
layer = resnet_block(layer, 128, 3, 0, 'relu', cross_block=True, shrink=True)
layer = resnet_block(layer, 128, 3, 0, 'relu')

layer = resnet_block(layer, 256, 3, 0, 'relu', cross_block=True, shrink=True)
layer = resnet_block(layer, 256, 3, 0, 'relu')
layer = Flatten()(layer)
output = Dense(units=1, activation='sigmoid')(layer)

model = Model(inputs=[input], outputs=[output])

# input = Input(shape=(56,))
# layer1 = Dense(64,activation='relu')(input)
# layer2 = Dense(128,activation='relu')(layer1)
# output = Dense(2,activation='softmax')(layer2)
#
# model = Model(input = input, output = output)
optimizer = RMSprop(0.001)

model.summary()
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_squared_error'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, mode='auto', restore_best_weights=True)

model.fit(dataset.train_set['x'], dataset.train_set['y'],
          batch_size=500,
          epochs=1000,
          validation_data=(dataset.val_set['x'], dataset.val_set['y']),
          callbacks=[monitor])

model.save('model_%s_2'%device_type)

# --------------------validation-------------------------
pred = model.predict(dataset.test_set['x'])
results = pred
threshold = 0.9
results[results >= threshold] = 1
results[results < threshold] = 0

cm = cm_metrix(dataset.test_set['y'], results)

cm_analysis(cm, ['Normal', 'malfunction'], precision=True)