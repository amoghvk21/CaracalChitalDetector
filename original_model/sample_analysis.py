import matplotlib.pyplot as plt
import numpy as np
import pickle
import librosa
import random
import pytz
from datetime import datetime


WORKING_DIR = 'C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/'


# Calculated in remove_noise.py
MEAN_C_CALL_FREQ_HIGH = 1822.8
MEAN_C_CALL_FREQ_LOW = 765.7
MEAN_DELTA_TIME = 1.534

def get_5_random(d):
        # Create list of all
        all_items = []
        for (utctimestamp, deviceno), times in d.items():
            for time in times:
                all_items.append((utctimestamp, deviceno, time))

        # Get 5 random
        random.shuffle(all_items)
        all_items = all_items[:5]
        
        return all_items


def get_items():
    # Get 5 random TPs/FPs for analysis
    with open(WORKING_DIR + 'data/py_obj/TPs_templating.pkl', 'rb') as f:
        TPs_templating = pickle.load(f)

    with open(WORKING_DIR + 'data/py_obj/FPs_templating_new.pkl', 'rb') as f:
        FPs_templating = pickle.load(f)

    print('TP:')
    TPs = get_5_random(TPs_templating)
    #TPs = [(1711540800, 213, 1185.088), (1711540800, 213, 1461.056), (1711540800, 204, 1180.512), (1711540800, 204, 1143.136), (1711458000, 217, 3140.576)]
    print()
    print('FPs:')
    FPs = get_5_random(FPs_templating)

    with open(WORKING_DIR + "data/py_obj/TP_FP_5_sample.pkl", 'wb') as f:
        pickle.dump((TPs, FPs), f)


def retrieve_items():
    with open(WORKING_DIR + "data/py_obj/TP_FP_5_sample.pkl", 'rb') as f:
        return pickle.load(f)


def display_items(all_items):
    # Table headings
    print('i\tdevice no\tdate\t\t\t\ttime')

    for i, (utctimestamp, deviceno, time) in enumerate(all_items):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        
        # Print into
        print(f'{i+1}\t{deviceno}\t\t{dt}\t{time}')
        
        # Only 5 items (Used when the sample was larger). As only 5 items anyway, code is redundant
        if i == 5:
            break
    print()
    

def get_time_bounds_TP(items):
    with open(WORKING_DIR + 'data/py_obj/selectiontabledata.pkl', 'rb') as f:
        data = pickle.load(f)

    # List of (utctimestamp, deviceno, LB, UB)
    new_items = []

    # Go through each item and find the bounding box it is inside
    for utctimestamp, deviceno, time in items:
        times = data[utctimestamp, deviceno]
        for LB, UB, species in times:
            # Check if inside the bounding box
            if LB < time and UB > time and species == 'C':
                break
        new_items.append((utctimestamp, deviceno, LB, UB))
    
    return new_items


def get_time_bounds_FP(items):
    new_items = []

    for utctimestamp, deviceno, time in items:
        new_items.append((utctimestamp, deviceno, round(time-MEAN_DELTA_TIME, 3), round(time+MEAN_DELTA_TIME, 3)))

    return new_items


def generate_plots(items, type):

    all_mfccs = []
    all_mean_mfccs = []
    
    for i, (utctimestamp, deviceno, LB_time, UB_time) in enumerate(items):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that audio
        chopped_aud = aud[LB_sample:UB_sample]

        # Calcuate MFCC for this chhopped audio clip between the relevant frequencies and append to a list
        mfcc = librosa.feature.mfcc(y=chopped_aud, sr=sr, fmin=MEAN_C_CALL_FREQ_LOW, fmax=MEAN_C_CALL_FREQ_HIGH, n_mfcc=12)
        all_mfccs.append(mfcc)

        # PLOT 1:
        # Heatmaps for each call's MFCCs
        # Create plot
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'MFCC Heatmap of Type: {type} | UTC: {utctimestamp} | deviceno: {deviceno} | Time: {LB_time} to {UB_time}')
        plt.ylabel("MFCC Coefficients")
        plt.savefig(WORKING_DIR + f'plots/heatmaps/{type}_{i}_heatmap_mfcc_{utctimestamp}_{deviceno}_{str(LB_time)}_{str(UB_time)}.jpeg', format='jpeg')
        
        # PLOT 2:
        # Bar chart of the mean mfcc feature for all time
        mean_mfcc = []
        for mfccs in mfcc:
            mean_mfcc.append(np.mean(mfccs))
        all_mean_mfccs.append(mean_mfcc)
        
        plt.figure(figsize=(10, 4))
        plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], mean_mfcc)
        plt.title(f'Barchart of mean MFCC of Type: {type} | UTC: {utctimestamp} | deviceno: {deviceno} | Time: {LB_time} to {UB_time}')
        plt.xlabel("MFCC Coefficients")
        plt.ylabel("Mean Magnitude of the Coefficient")
        plt.savefig(WORKING_DIR + f'plots/barcharts/{type}_{i}_barchart_mean_mfcc_{utctimestamp}_{deviceno}_{str(LB_time)}_{str(UB_time)}.jpeg', format='jpeg')

        print(f'{type}_{i}')

    # PLOT 3 and 4:
    # Multiple bar chart for mean of each coefficient for all 5 samples of TP and FP

    # Normalise all coefficients so that you can compare them with each other
    all_mean_mfccs = np.array(all_mean_mfccs)
    all_mean_mfccs = all_mean_mfccs.T
    new = np.array([x/(np.sum(np.abs(x))) for x in all_mean_mfccs])
    print(new[0])
    all_mean_mfccs = new.T

    plt.figure(figsize=(10, 4))
    r = np.arange(12) 
    width = 0.1
    for i in range(5):
        plt.bar(r+(width*i), all_mean_mfccs[i], width=width, label=i) 
    plt.xticks(r + 5*width/2, ['1','2','3','4', '5', '6', '7', '8', '9', '10', '11', '12']) 
    plt.legend()
    plt.xlabel("MFCC Coefficients")
    plt.ylabel("Mean Magnitude of the Coefficients")
    plt.title(f'Mean barcharts for all {type}s')
    plt.savefig(WORKING_DIR + f'plots/{type}_multiplebarchart_mfcc_mean.jpeg', format='jpeg')

    return all_mean_mfccs


def generate_plots_2(all_mean_mfccs_TP, all_mean_mfccs_FP):
    
    # PLOT 5:
    # Take mean of average mfcc value for TP and FP and look at each side by side
    all_mean_mfccs_TP = np.array([np.array(x) for x in all_mean_mfccs_TP])
    all_mean_mfccs_FP = np.array([np.array(x) for x in all_mean_mfccs_FP])

    all_mean_mfccs_TP = all_mean_mfccs_TP.T
    all_mean_mfccs_FP = all_mean_mfccs_FP.T
    
    y_TP = [np.mean(y) for y in all_mean_mfccs_TP]
    y_FP = [np.mean(y) for y in all_mean_mfccs_FP]

    y = [y_TP, y_FP]
    label = ['TP', 'FP']

    plt.figure(figsize=(10, 4))
    r = np.arange(12) 
    width = 0.1
    for i in range(2):
        plt.bar(r+(width*i), y[i], width=width, label=label[i]) 
    plt.xticks(r + 2*width/2, ['1','2','3','4', '5', '6', '7', '8', '9', '10', '11', '12']) 
    plt.legend()
    plt.xlabel("MFCC Coefficients")
    plt.ylabel("Mean Magnitude of the Coefficients")
    plt.title("Mean barchart for TPs and FPs")
    plt.savefig(WORKING_DIR + f'plots/multiplebarchart_mfcc_mean_2.jpeg', format='jpeg')


def main():
    #get_items()  # Comment out once you have got and saved your 10 items for analysis

    TPs, FPs = retrieve_items()

    print("TPs:")
    display_items(TPs)
    print("FPs:")
    display_items(FPs)

    TP_times = get_time_bounds_TP(TPs)
    FP_times = get_time_bounds_FP(FPs)

    meanTP = generate_plots(TP_times, 'TP')
    meanFP = generate_plots(FP_times, 'FP')

    generate_plots_2(meanTP, meanFP)

if __name__ == '__main__':
    main()