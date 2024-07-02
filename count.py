# Used to test testbed.py
import pickle


# Just counts no of calls which should be the same as TP + FN
with open('selectiontabledata.pkl', 'rb') as f:
    actual = pickle.load(f)

count = 0

for (_, _), times in actual.items():
    for _, _, species in times:
        if species == 'C':
            count += 1

print(f'TP + FN (no of calls): {count}')


# Just counts total no of detections which should be the same as TP + FP
# This case we are testing the templating model
# Counts no of deciaml points (as each detection has one) and minuses the decimal points from the .wav part as they don't correspond to a detection
count = 0

with open('autodetect.txt', 'r') as f:
    for line in f:
        count += line.count('.')

count -= 26

print(f'TP + FP (no of detections): {count}')