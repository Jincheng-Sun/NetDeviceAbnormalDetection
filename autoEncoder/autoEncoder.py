from keras.models import load_model, save_model
from keras import Model
from keras.layers import Dense, Input
import numpy as np

# -------------------------------------------------------------

'''Using keras to build the autoencoder model'''
origin_feature = Input(shape=(56,))
encode = Dense(20, activation='relu')(origin_feature)
decode = Dense(56, activation='tanh')(encode)

autoencoder = Model(input=origin_feature, output=decode)
'''Share wights of the first layer'''
encoder = Model(input=origin_feature, output=encode)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# -------------------------------------------------------------

file_path = 'preconcat'

'''load data'''
x_train1 = np.load('../data/%s/real_alarm_x.npy' % file_path)
x_train2 = np.load('../data/%s/fake_normal_x.npy' % file_path)
x_train3 = np.load('../data/%s/real_normal_x.npy' % file_path)
x_train = np.concatenate((x_train1, x_train2, x_train3), axis=0)
del x_train1
del x_train2
del x_train3

from sklearn.model_selection import train_test_split
x_train, x_val = train_test_split(x_train, test_size=0.1, random_state=42)

# -------------------------------------------------------------

from keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
autoencoder.fit(x=x_train, y=x_train,
                epochs=100,
                batch_size=256,
                validation_data=(x_val, x_val),
                callbacks=[callback]
                )

save_model(autoencoder, '../models/%s/autoencoder'%file_path)
save_model(encoder, '../models/%s/encoder'%file_path)

# -------------------------------------------------------------
#
# autoencoder = load_model('../models/autoencoder')
# encoded_input = Input(shape=(10,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
#
# encoder = load_model('../models/encoder')
# mal_x = np.load('../data/mal_x.npy')
# # x = mal_x.reshape([-1,45])[1]
# x_ = encoder.predict(mal_x.reshape([-1, 56]))
# np.save('data/d20_mal_x.npy', x_)
