import pysptk
import numpy as np

# Run this step first and save features so that we don't need do to extraction every time we train

def extract_mfccs(waveforms, settings, mode=None):
    all_mfccs = []
    padded_mfccs = []
    all_lengths = []
    for waveform in waveforms:
        mfccs = []
        fs = waveform[0]
        x = waveform[1]
        for pos in range(0, (len(x)-settings.frame_step), settings.frame_step):
            xslice = x[pos:pos+settings.frame_length]
            if len(xslice) < settings.frame_length:
                padding = [0 for n in list(range(0,settings.frame_length-len(xslice)))]
                xslice = np.concatenate((xslice, np.array(padding)))
            xw = xslice * pysptk.blackman(settings.frame_length)
            ext_mfcc = pysptk.sptk.mfcc(xw, fs=fs, order=24, num_filterbanks=24*2)
            mfccs.append(ext_mfcc)

        # Accumulate sentence level mfccs and length
        all_lengths.append(len(mfccs))
        all_mfccs.append(mfccs)

    #print 'mfccs sizes', all_lengths # To check upsample_alignment
    max_T = max(all_lengths)+2 # just in case
    # Do padding given max_T
    padded_mfccs = [np.pad(np.array(m), ((0, max_T-np.array(m).shape[0]), (0, 0)), 'constant') for m in all_mfccs]
    return padded_mfccs, max_T, all_lengths
