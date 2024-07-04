# Used to test testbed.py
import pickle


WORKING_DIR = 'C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/'


# Just counts no of calls which should be the same as TP + FN
with open(WORKING_DIR + 'data/py_obj/selectiontabledata.pkl', 'rb') as f:
    actual = pickle.load(f)

count = 0

# Going through each selection table and counting all the C's
# Number of actual C calls which is TP + FN
for (_, _), times in actual.items():
    for _, _, species in times:
        if species == 'C':
            count += 1

print(f'TP + FN (no of calls): {count}')


# Just counts total no of detections which should be the same as TP + FP
# This case we are testing the templating model
# Counts no of deciaml points (as each detection has one) and minuses the decimal points from the .wav part as they don't correspond to a detection

# WRONG - As TP + FP is not equal to no of detections 
# due to some FPs corresponding to already found calls and they are ignored as FPs in testbed.py
count = 0

with open(WORKING_DIR + 'original_model/autodetect.txt', 'r') as f:
    for line in f:
        count += line.count('.')

count -= 26

print(f'TP + FP (no of detections): {count}')