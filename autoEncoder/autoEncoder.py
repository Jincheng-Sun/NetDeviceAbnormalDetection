from keras.models import load_model,save_model
from keras import Model
from keras.layers import Dense, Input
import numpy as np
input = Input(shape = (56,))

encode = Dense(20,activation='relu')(input)
decode = Dense(56,activation='tanh')(encode)

autoencoder = Model(input = input, output = decode)
encoder = Model(input = input, output = encode)

autoencoder.compile(optimizer='adam', loss='mse')



x_train1 = np.load('../data/mal_x.npy')
x_train2 = np.load('../data/nor_x.npy')
x_train = np.concatenate((x_train1,x_train2),axis=0)
del x_train1
del x_train2
autoencoder.fit(x_train, x_train,
                nb_epoch=5,
                batch_size=256,
                shuffle=True)

save_model(autoencoder,'../models/autoencoder')
save_model(encoder,'../models/encoder')
autoencoder = load_model('../models/autoencoder')
encoded_input = Input(shape=(10,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

encoder = load_model('../models/encoder')
mal_x = np.load('../data/mal_x.npy')
# x = mal_x.reshape([-1,45])[1]
x_ = encoder.predict(mal_x.reshape([-1,56]))
np.save('data/d20_mal_x.npy',x_)

