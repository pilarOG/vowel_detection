import os
from scipy.io import wavfile
import codecs


def load_waves(settings, file_list):
    # List of (fs, x) tuples
    wavefiles = [wavfile.read(settings.wav_folder+'/'+filepath) for filepath in file_list]
    return wavefiles

def load_labels(settings, file_list):
    alignments = [codecs.open(settings.label_folder+'/'+filepath.replace('wav', 'lab'), encoding='utf-8').read().split('\n') for filepath in file_list]
    return alignments
