
import re
import math
import random
import numpy as np

# We have to upsample alignments to the frame level to match the dimensions of the mfccs
# Right now only HTK like labels
def upsample_alignment(alignments, max_T, settings, mode='htk'):
    label_windows = []
    label_lengths = []
    label_targets = []
    for j in range(0, len(alignments)):
        label = alignments[j]
        frames_per_phone = []
        sentence_phones = []
        windows_per_phone = []
        sentence_frames = 0
        sentence_duration = 0
        sentence_windows = 0
        for line in label:
            if 'score' in line:
                if mode == 'htk':
                    is_vowel = [0]
                    timestamp = float(re.findall(r'\t(.*?)\t', line)[0])
                    phone = re.findall(r'\t.*\t(.*?)\s\;\sscore', line)[0]
                    #print timestamp, phone
                    # Check if it's vowel
                    if phone in settings.vowel_list: is_vowel = [1]
                    # Calculate seconds it takes
                    duration = timestamp - sentence_duration
                    # Calculate frames
                    frames = (duration * float(settings.samplerate))
                    #print phone, duration, int(frames)
                    sentence_frames += int(frames)
                    sentence_duration += duration
                    sentence_windows += (frames/512)*2
                    frames_per_phone.append(frames)
                    sentence_phones.append(is_vowel)
                    windows_per_phone.append((frames/512)*2)

        label_windows.append(int(sentence_windows))

        # multiply bool by ints
        targets = []
        for i in range(0, len(sentence_phones)):
            #print sentence_phones[i], windows_per_phone[i]
            for _ in range(0, int(windows_per_phone[i])+random.choice([0,0,0,1])): # How do they correct for this in Merlin?
                targets.append(sentence_phones[i])
        if len(targets) < max_T:
            padded_targets = targets + [[0] for n in list(range(0,max_T-len(targets)))]
        label_targets.append(padded_targets)
        label_lengths.append(len(padded_targets))

    print 'label sizes', label_windows
    #print 'target sizes', label_lengths
    return label_targets
