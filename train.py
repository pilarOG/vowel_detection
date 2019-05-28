# -*- coding: utf-8 -*-

# Train a model to classify each frame as vowel/not
# After classification find the peak intensity on consecutive frames classified as vowels
# Add marks at the chosen points and calculate speech rate using them

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from get_labels import upsample_alignment
from load_data import load_waves, load_labels
from configure import load_config
from argparse import ArgumentParser
from extract_features import extract_mfccs
import numpy as np
import os
from utils import plot_loss_acc
from sklearn.utils import shuffle
from keras.utils import to_categorical
from net_architectures import build_feedforward
### MAIN ###

a = ArgumentParser()
a.add_argument('-c', dest='config', required=True, type=str)
opts = a.parse_args()
settings = load_config(opts.config)

# Load training data
file_list = os.listdir(settings.wav_folder)
#TODO: check that there are only wave files
waveforms = load_waves(settings, file_list)
mfccs, max_T, all_lengths = extract_mfccs(waveforms, settings) #TODO: what other features? intensity? HNR?
# print mfccs.shape # (number of files, max_T, 12 coeff)
alignments = load_labels(settings, file_list)
targets = upsample_alignment(alignments, max_T, settings)
print ('mfccs shape', np.array(mfccs).shape)
print ('targets shape', np.array(targets).shape)

# x = mfccs and y = labels
x = np.array(mfccs)
y = np.array(targets)

x_fbf = []
y_fbf = []
c_1 = 0
c_0 = 0

# Count number of data points with targets equal 1s and 0s, balance data if unbalanced
num_ones = (y[:,:,0] == 1).sum()
num_zeros = (y[:,:,0] == 0).sum()
print ('number of 1s', num_ones)
print ('number of 0s', num_zeros)
if num_zeros != num_ones:
    #print ('data unbalanced, balancing data...')
    if num_zeros > num_ones:
        num_balance_data = num_ones
    else:
        num_balance_data = num_zeros
for sample in range(0, x.shape[0]):
    for frame in range(0, all_lengths[sample]): # can we avoid padding?
        if y[sample,frame,:][0] == 1 and c_1 < num_balance_data: # balance classes
            c_1 += 1
            x_fbf.append((x[sample,frame,:]))
            y_fbf.append((y[sample,frame,:]))
        elif y[sample,frame,:][0] == 0 and c_0 < num_balance_data: # balance classes
            c_0 += 1
            x_fbf.append((x[sample,frame,:]))
            y_fbf.append((y[sample,frame,:]))
print ('new number of 1s and 0s:', c_1, c_0)

# Silicing data in training, validation and test sets
test_p = settings.test_p
val_p = settings.val_p
print ('slicing dataset into train, val and test sets')
print ('val percentage: ', val_p)
print ('test percentage: ', test_p)

total_samples = np.array(x_fbf).shape[0]
test_samples = int(float(total_samples) * float(test_p) // float(100)) # calculate number of test samples
val_samples = int(float(total_samples) * float(val_p) // float(100)) # calculate number of validation samples
train_samples = total_samples-(test_samples+val_samples)
y_fbf = to_categorical(y_fbf) # change to categorical values
x_fbf_sh, y_fbf_sh = shuffle(x_fbf, y_fbf) # shuffle samples

x_train = np.array(x_fbf_sh)[0:train_samples,:]
y_train = np.array(y_fbf_sh)[0:train_samples,:]
x_val = np.array(x_fbf_sh)[train_samples:train_samples+val_samples,:]
y_val = np.array(y_fbf_sh)[train_samples:train_samples+val_samples,:]
x_test = np.array(x_fbf_sh)[train_samples+val_samples:train_samples+val_samples+test_samples,:]
y_test = np.array(y_fbf_sh)[train_samples+val_samples:train_samples+val_samples+test_samples,:]
print ('sets dimensions: train x, train y, val x, val y, test x, test y')
print (x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)


# Hyperparameters
batch_size = settings.batch_size
dropout_rate = settings.dropout_rate
n_feats = x_train.shape[1]
epochs = settings.epochs
nn_type = settings.nn_type # feed_forward only one supported right now

# build model
model = build_feedforward(n_feats, dropout_rate)
# train model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, verbose=2, batch_size=batch_size)
# evaluate model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
plot_loss_acc(history, settings.model_name)

# serialize model to JSON
model_json = model.to_json()
with open(settings.model_name+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(settings.model_name+".h5")
print("saved model")
