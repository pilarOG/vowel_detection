from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers

def build_feedforward(n_feats, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Dense(124, input_dim=n_feats, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(124, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='hard_sigmoid'))
    # loss and optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
