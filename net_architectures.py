from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_feedforward(n_feats, dropout_rate):
    model = Sequential()
    model.add(Dense(36, input_dim=n_feats, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(36, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='hard_sigmoid'))
    # loss and optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
