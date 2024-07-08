from glob import glob
import pickle
import datetime
from collections import defaultdict
import numpy as np
import os


WORKING_DIR = 'C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/'
OFFSET = 0.5

def create_master_db():
    paths = glob(WORKING_DIR + "data/Test set/1 hour files/*.txt", recursive=True)
    
    data = defaultdict(lambda: [])

    for path in paths:
        with open(path, 'r') as f:
            filename = path[115:]
            
            # Getting deviceno from path as it can be variable length
            deviceno = ''
            i = 3
            while filename[i] != '_':
                deviceno += filename[i]
                i += 1
            deviceno = int(deviceno)
            localtimestamp = datetime.datetime(int(filename[i+1:i+5]), int(filename[i+5:i+7]), int(filename[i+7:i+9]), int(filename[i+10:i+12]), int(filename[i+12:i+14]), int(filename[i+14:i+16]))
            utctimestamp = int(filename[i+17:i+27])
            
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
                    begintime = round(float(line[4]), 3)
                    endtime = round(float(line[5]), 3)
                    species = line[-1]

                data[(utctimestamp, deviceno)].append((begintime, endtime, species))

    # Save parsed data for easier access later
    with open(WORKING_DIR + 'data/py_obj/selectiontabledata.pkl', 'wb') as f:
        pickle.dump(dict(data), f)


def testbed(preds):
    with open(WORKING_DIR + "data/py_obj/selectiontabledata.pkl", 'rb') as f:
        actual = pickle.load(f)

    TP = 0
    FP = 0
    FN = 0

    found = set() # Used to make sure you don't double count and used to count FPs

    # Used in removenoise.py for reducing amount of FPs
    FPs_templating = defaultdict(lambda: [])
    TPs_templating = defaultdict(lambda: [])

    # Used to store all the FPs that refer to Os
    found_O = set()
    found_O_dict = defaultdict(lambda: [])
    
    # Matching each prediction with the actual call (cheking if inside bounding box) to count TP
    # If we cannot find an actual call for a prediction, then must be a FP
    # <variable> is referring to preds
    # <variable>1 is referring to actual
    for (utctimestamp, deviceno), times in preds.items():

        # Loops through every prediction
        for time in times:
            times1 = actual[(utctimestamp, deviceno)]  # Getting the actual calls for that file
            foundpred = False  # Flag used to count FPs when no prediction isn't inside any actual timebox

            # Loops through each actual time box and checks if prediction is inside one of them
            for (begintime1, endtime1, species1) in times1:

                # Checks if time is within the margin between begintime1 and endtime1 and if correct species
                # Checks if this call hasn't already been found - makes sure not counting multiple detections within a single margin
                if begintime1-OFFSET < time and endtime1+OFFSET > time and species1 == 'C' and (utctimestamp, deviceno, begintime1, endtime1) not in found:
                    TP += 1
                    found.add((utctimestamp, deviceno, begintime1, endtime1))
                    foundpred = True
                    TPs_templating[(utctimestamp, deviceno)].append(time) # Used for analysis later
                    break
                
                # Stores all the FPs that refer to Os - Used in different file
                elif begintime1 < time and endtime1 > time and species1 == 'O' and (utctimestamp, deviceno, begintime1, endtime1) not in found_O:
                    found_O.add((utctimestamp, deviceno, begintime1, endtime1))
                    found_O_dict[(utctimestamp, deviceno)].append((begintime1, endtime1))
                    break
                
                # If detection is already found, then we don't add it to FPs and we just ignore it
                elif begintime1 < time and endtime1 > time and species1 == 'C' and (utctimestamp, deviceno, begintime1, endtime1) in found:
                    foundpred = True
                    break
                

            if not foundpred:
                FP += 1
                FPs_templating[(utctimestamp, deviceno)].append(time)  # Used to save for analysis later

            foundpred = False

    #print('already found:', already_found)

    with open(WORKING_DIR + "data/py_obj/FPs_templating.pkl", 'wb') as f:
        pickle.dump(dict(FPs_templating), f)


    with open(WORKING_DIR + "data/py_obj/TPs_templating.pkl", 'wb') as f:
        pickle.dump(dict(TPs_templating), f)
    
    with open(WORKING_DIR + "data/py_obj/FPs_templating_O.pkl", 'wb') as f:
        pickle.dump(dict(found_O_dict), f)

    # Go through each call in actual and count calls that werent found by the model and so not in 'found'. These are FN
    # <variable>1 is referring to actual
    for (utctimestamp1, deviceno1), d in actual.items():
        for begintime1, endtime1, species1 in d:
            if (utctimestamp1, deviceno1, begintime1, endtime1) not in found and species1 == 'C':
                FN += 1

    FP = 281   # After remove_noise.py
    
    return TP, FP, FN


# CHANGE IMPLEMENTATION BASED ON WHICH MODEL YOU WANT TO TEST
# At the moment, testing pre existing templating model
def get_preds():

    preds = {}

    with open(WORKING_DIR + 'original_model/autodetect.txt', 'r') as f:
        for line in f.readlines():
            line = line.split()
            path = line[0]

            # Getting deviceno from path as it can be variable length
            deviceno = ''
            i = 3
            while path[i] != '_':
                deviceno += path[i]
                i += 1
            deviceno = int(deviceno)
            utctimestamp = int(path[i+17:i+27])

            # Converts all times to floats
            preds[utctimestamp, deviceno] = [float(t) for t in line[1:]]

    # Used to save for analysis later
    '''
    with open(WORKING_DIR + "data/py_obj/templatingresult.pkl", 'wb') as f:
        pickle.dump(preds, f)
    '''

    return preds


def main():
    create_master_db()
    preds = get_preds()
    TP, FP, FN = testbed(preds)

    print(f'TP:\t\t\t\t\t{TP}')
    print(f'FP:\t\t\t\t\t{FP}')
    print(f'FN:\t\t\t\t\t{FN}')
    print()
    print(f'TP + FN (Total no of calls):\t\t{TP+FN}')
    print(f'TP + FP (Total no of detections):\t{TP+FP}')
    print()
    print(f'TP/Total no of calls:\t\t\t{TP/(TP+FN)}')
    print(f'FP/Total no of detections:\t\t{FP/(TP+FP)}')


if __name__ == "__main__":
    main()