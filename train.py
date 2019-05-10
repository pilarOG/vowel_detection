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

### MAIN ###

a = ArgumentParser()
a.add_argument('-c', dest='config', required=True, type=str)
opts = a.parse_args()
settings = load_config(opts.config)

# Load training data
file_list = os.listdir(settings.wav_folder)
#TODO: check that there are only wave files
waveforms = load_waves(settings, file_list)
mfccs, max_T = extract_mfccs(waveforms, settings)
# print mfccs.shape # (number of files, max_T, 12 coeff)
alignments = load_labels(settings, file_list)
targets = upsample_alignment(alignments, max_T, settings)
print ('mfccs shape', np.array(mfccs).shape)
print ('targets shape', np.array(targets).shape)

x_train = np.array(mfccs)
y_train = np.array(targets)
