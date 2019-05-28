from configure import load_config
from argparse import ArgumentParser
from load_data import load_waves
from extract_features import extract_mfccs
from keras.models import model_from_json
import numpy as np
import os

### Then the main use would be:
# 1) take the frames from a sentence
# 2) pass them through the model
# 3) Find the classes
# 4) Find where vowels start and end
# 5) make a guess of how many vowels are in the sentence

a = ArgumentParser()
a.add_argument('-c', dest='config', required=True, type=str)
opts = a.parse_args()
settings = load_config(opts.config)

# Load test data
file_list = os.listdir(settings.wavs_test)
waveforms = load_waves(settings, file_list)
# 1) extract mfccs
mfccs, _, all_lengths = extract_mfccs(waveforms, settings)
x_test = np.array(mfccs)

# load model
json_file = open(settings.model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(settings.model_name+".h5")
print("loaded model")

# 2) analyse frames
print (x_test.shape)
for sentence in range(0, x_test.shape[0]):
    current_sentence = x_test[sentence,:,:]
    trim_current_sentence = current_sentence[0:all_lengths[sentence],:]
    predictions = loaded_model.predict_classes(trim_current_sentence, batch_size=32)
    print ('new_sentence', file_list[sentence])
    sample_t = 0
    for frame in predictions:
        print (sample_t, frame)
        time = float(settings.frame_length-settings.frame_step) / float(settings.samplerate)
        sample_t += time
