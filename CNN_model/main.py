# Get counter and count how large the ararys will be be adding len(chopped aud) to the counter and constantly printing

import numpy as np
import tensorflow as tf
import librosa
import pickle
from collections import defaultdict
from glob import glob
import datetime


WORKING_DIR = 'C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/'


with open(WORKING_DIR + 'data/py_obj/selectiontabledata.pkl', 'rb') as f:
    true_raw_data = pickle.load(f)

# Go through all files, get each true call by trimming based on selection table data 
# Save results
paths = glob(WORKING_DIR + "data/Test set/1 hour files/*.txt", recursive=True)

all_calls = defaultdict(lambda: [])
for j, path in enumerate(paths):
    filename = path[115:]
    
    # Getting deviceno from path as it can be variable length
    deviceno = ''
    i = 3
    while filename[i] != '_':
        deviceno += filename[i]
        i += 1
    deviceno = int(deviceno)
    localtimestamp = dt = datetime.datetime(int(filename[i+1:i+5]), int(filename[i+5:i+7]), int(filename[i+7:i+9]), int(filename[i+10:i+12]), int(filename[i+12:i+14]), int(filename[i+14:i+16]))
    utctimestamp = int(filename[i+17:i+27])

    print(j)

    if dt.year < 2024:
        continue

    # Look in raw data and get the times
    times = true_raw_data[(utctimestamp, deviceno)]

    # Load the audio
    for (begintime, endtime, species) in times:

        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        LB_sample = round(begintime*sr)
        UB_sample = round(endtime*sr)

        # For each time chop the audio and add to all_calls
        chopped_aud = aud[LB_sample:UB_sample]
        all_calls[(utctimestamp, deviceno)].append(chopped_aud)

# Store all_calls for later use
with open(WORKING_DIR + 'data/py_obj/all_calls.pkl', 'wb') as f:
    pickle.dump(all_calls, f)