from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, add, Input, Dense, Flatten
from keras import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


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
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=strides,
                          padding='same')(layer)
    else:
        shortcut = layer

    if not is_first:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=strides,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=1,
                   padding='same')(layer)
    layer = add([shortcut, layer])

    if is_last:
        layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    return layer

'''create model'''
input = Input(shape=(1, shape))
layer = Conv2D(filters=32,
               kernel_size=3,
               kernel_initializer='random_uniform',
               # kernel_regularizer=regularizers.l2(0.01),
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
output = Dense(units=output_num, activation='softmax')(layer)

model = Model(inputs=[input], outputs=[output])

optimizer = Adam(lr=0.005)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=3, mode='auto', restore_best_weights=True)

model.fit(train_x, train_y,
          batch_size=50,
          epochs=1000,
          validation_split=0.1,
          callbacks=[monitor])
