import pickle
import librosa
import numpy as np
from datetime import datetime
import pytz
from glob import glob
from collections import defaultdict


# Relative working directory
WORKING_DIR = 'C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/'

# Temp list of all data before calculating mean
high_freqs = []
low_freqs = []
delta_times = []

paths = glob(WORKING_DIR + "data/Test set/1 hour files/*.txt", recursive=True)

for path in paths:
    with open(path, 'r') as f:
        filename = path[110:]
        
        # Getting deviceno from path as it can be variable length
        deviceno = ''
        i = 3
        while filename[i] != '_':
            deviceno += filename[i]
            i += 1
        localtimestamp = datetime(int(filename[i+1:i+5]), int(filename[i+5:i+7]), int(filename[i+7:i+9]), int(filename[i+10:i+12]), int(filename[i+12:i+14]), int(filename[i+14:i+16]))
        
        # Not using 2023 files as badly labelled
        if localtimestamp.year < 2024:
            continue

        for line in f.readlines():
            # Ignore first line of column names
            if line[0] == 'S':
                continue

            line = line.split()
            
            # Ignore anything that isnt a spectogram record
            if line[1] != 'Spectrogram':
                continue
            else:
                # Rounding times to 3 dp
                low_freqs.append(float(line[6]))
                high_freqs.append(float(line[7]))
                delta_times.append(round(float(line[5])-float(line[4]), 3))

# Upper and lower bound of typical frequencies of bird calls
# Work out mean low and high freq
MEAN_C_CALL_FREQ_LOW = round(np.mean(low_freqs), 1)
MEAN_C_CALL_FREQ_HIGH = round(np.mean(high_freqs), 1)

# How much left and right of the detection should we look and find variance
# Work out mean delta time
MEAN_DELTA_TIME = round(np.mean(delta_times), 3)


#####################################################################################


sd_TP = []
sd_FP = []


# Load the TPs to calculate stats about it 
with open(WORKING_DIR + "data/py_obj/TPs_templating.pkl", 'rb') as f:
    TPs_templating = pickle.load(f)

for (utctimestamp, deviceno), times in TPs_templating.items():
    # Get filepath and open wav file
    timezone = pytz.timezone('Asia/Kathmandu')
    dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
    aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

    # Go through each call
    for time in times:
        # Get lower and upper bound of the time duration of the detection
        LB_time = time - MEAN_DELTA_TIME
        UB_time = time + MEAN_DELTA_TIME
        
        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that detection
        chopped_aud = aud[LB_sample:UB_sample]

        # Parameters for generating spectogram
        n_fft = 1024
        hop_length = 512
        n_mels = 64       

        mel_spec = librosa.feature.melspectrogram(y=chopped_aud, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Convert to dB values
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Flatten matrix into single list of vaules
        flat_mel_spec_db = mel_spec_db.flatten()

        # Append sd of the dB values for that detection
        sd_TP.append(np.std(flat_mel_spec_db))

# Calculate mean of all the sd of all dB values of all detections
# Used to distinguish between noise and an actual detection
sd_TP_mean = np.mean(sd_TP)


# Data structure containing info about FPs before this script is ran
with open(WORKING_DIR + "data/py_obj/FPs_templating.pkl", 'rb') as f:
    FPs_templating = pickle.load(f)

# No of FPs that will now be classified as TN due to being random noise
total = 0

# Previous no of FPs
prev_total = 0

# New dict of FPs after removed the ones with noise
new_FPs = defaultdict(lambda: [])

for (utctimestamp, deviceno), times in FPs_templating.items():
    # Get filepath and open wav file
    timezone = pytz.timezone('Asia/Kathmandu')
    dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
    aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

    # Go through each detection
    for time in times:
        # Get lower and upper bound of the time duration of the detection
        LB_time = time - MEAN_DELTA_TIME
        UB_time = time + MEAN_DELTA_TIME
        
        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that detection
        chopped_aud = aud[LB_sample:UB_sample]

        # Parameters for generating spectogram
        n_fft = 1024
        hop_length = 512
        n_mels = 64        

        mel_spec = librosa.feature.melspectrogram(y=chopped_aud, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Convert to dB values
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Flatten matrix into single list of vaules
        flat_mel_spec_db = mel_spec_db.flatten()

        # Calculate sd of this detection
        sd = np.std(flat_mel_spec_db)

        # Check if calculated sd is less than mean, then will be classified as noise
        if sd < sd_TP_mean:
            total += 1
        else:
            # These are still FPs so add them to the new datastructure storing this reduced set of FPs
            new_FPs[(utctimestamp, deviceno)].append(time)
            
        
        prev_total += 1

# Used in analysis of a sample
with open(WORKING_DIR + "data/py_obj/FPs_templating_new.pkl", 'wb') as f:
    pickle.dump(dict(new_FPs), f)

print(f'No of FP that can be changed to TN (due to detecting noise): {total}')
print(f'Previous no of FPs {prev_total}')
print(f'New no of FPs: {prev_total-total}')


# Resulting new FP is 502


# Count no of FPs that are O
# Load the actual data from selection tables
with open(WORKING_DIR + "data/py_obj/selectiontabledata.pkl", 'rb') as f:
    actual = pickle.load(f)

found = set()

# No of O calls in the new smaller set of FPs
O = 0

# Iterate through all these new FPs that we calculated before
for (utctimestamp, deviceno), times in new_FPs.items():

        # Loops through every FP
        for time in times:
            times1 = actual[(utctimestamp, deviceno)]    # Gets the actual calls for that file

            # Checks if a O call happened within according to the actual selection table data
            # Loops through each actual time box and checks if prediction is inside one of them
            for (begintime1, endtime1, species1) in times1:
                # Checks if time is within the margin between begintime1 and endtime1 and if correct species
                # Checks if this call hasn't already been found - makes sure not counting multiple detections within a single margin
                if begintime1 < time and endtime1 > time and species1 == 'O' and (utctimestamp, deviceno, begintime1, endtime1) not in found:
                    O += 1
                    found.add((utctimestamp, deviceno, begintime1, endtime1))

print()
print(f"No of O calls within the {prev_total-total} new FPs is: {O}")
print(f'New no of FPs is {prev_total-total-O}')