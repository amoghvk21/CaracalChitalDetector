import tensorflow as tf
import numpy as np
import librosa
import pickle
import time
import pandas
import datetime
import os
import keras
from keras import backend as K



ML_SR = 8000 # Target sampling rate
SPECD_FFT_LEN =  512 # Real FFT length (in the M4F - we use double of this on the PC as we don't do single-sided)
ML_BIN_AGG = 14 # Number of frequency bins (vertical dimension)
ML_FLO = 600 # Low freq
ML_FHI = 1400 # High freq
ML_FFT_STRIDE=1024 # Stride length in audio samples
ML_NUM_FFT_STRIDES = 12 # How many strides make up a sample
THRESHOLDED = False # Threshold the template or not

ONE_HOUR_FILE_PATH = "C:\\Users\\Amogh\\OneDrive - University of Cambridge\\Programming-New\\CaracalChitalDetector\\cnn\\data\\CAR204_20240325$164500_1711364400.wav"  # Which 1 hour file you want to look at
#SELECTION_TABLE_FILE_PATH = ONE_HOUR_FILE_PATH[:-3] + "Table.1.selections.txt"
#SELECTION_TABLE_FILE_PATH = SELECTION_TABLE_FILE_PATH.replace("data", "annotations")
#CNN_MODEL_PATH = "03MLouput20240811v001.keras"  # Path of where the pretrained model is stored
CNN_MODEL_PATH = "model_complex.keras"
CNN_OUTPUT_FILE_PATH = "output.pkl"  # Where the output of this model is stored - list of (LB, UB) times where a call is detected 
SELECTION_TABLE_OUTPUT = 'output.txt'  # Data from output.pkl stored in selection table format for analysis in raven
ANNOTATION_FILE = 'AcousticAnnotations001.pb'  # CSV file where all true annotations are stored
C_CALLS_ALL_DIR = 'c_calls_all.pkl' # Stores all the detections that are used



def chunkToBinsFixed(chunk,fLo,fHi,numbins,sr):
    """convert a chunk (window) to spectral image.
    Provide the low and high frequencies in Hz for a spectral windowing
    numbins are the number of output bins between flo and fhi
    Provide the sample rate in Hz"""
    CMPLX_FFT_LEN = len(chunk)*2
    fS = np.fft.fft(chunk,n=CMPLX_FFT_LEN) # fft - note we double it for the cmplx fft
    fRes = sr/(CMPLX_FFT_LEN)   # frequency per cmplx bin
    # Find the bin indices - map from frequency to FFT index
    binLo = int(fLo/sr*CMPLX_FFT_LEN)
    binHi = int(fHi/sr*CMPLX_FFT_LEN)
    specSize = int((binHi-binLo)/numbins)
    binTotals = np.zeros(numbins)
    for k in range(numbins):
        dbSum = 0
        for j in range(specSize):
            idx = binLo + (k * specSize) + j # NB not numbins!
            # Convert complex magnitude to absolute value
            absVal = np.abs(fS[idx])
            # We add an offset so we don't take log of tiny numbers. We can explore what a sensible offset is - 1.0 is probably too high.
            absVal += 1.0 
            # Convert to "power" by taking log
            dbVal = np.log(absVal) # NB natural (not log10) base!
            # Add up all the "powers" - again, this is probably not "correct", but we are just trying to work out some useful input features
            dbSum += dbVal
        binTotals[k] = dbSum
    return binTotals





# With random
def wavFileToSpecImg(aud,num_strides,target_sr=8000,random_offset=4000,THRESHOLDED=True,SCALED=True):
    tList = []
    startIdx = int(np.random.uniform(random_offset))
    for idx in range(0,num_strides*ML_FFT_STRIDE,int(ML_FFT_STRIDE)):
        clip = aud[idx+startIdx:idx+startIdx+SPECD_FFT_LEN]
        q = chunkToBinsFixed(clip,ML_FLO,ML_FHI,ML_BIN_AGG,ML_SR)
        tList.append(q)
    tList = np.array(tList)
    # Thresholding
    if THRESHOLDED:
        tList = (tList >0)*tList
    if SCALED:
        # Scale the dB mag spec to +1/-1
        maxVal = np.max(tList)
        minVal = np.min(tList)
        tList = (tList-minVal)/(maxVal-minVal)
    return np.array(tList)


# Without random
'''
def wavFileToSpecImg(aud,num_strides,target_sr=8000,THRESHOLDED=True,SCALED=True):
    tList = []
    for idx in range(0,num_strides*ML_FFT_STRIDE,int(ML_FFT_STRIDE)):
        clip = aud[idx:idx+SPECD_FFT_LEN]
        q = chunkToBinsFixed(clip,ML_FLO,ML_FHI,ML_BIN_AGG,ML_SR)
        tList.append(q)
    tList = np.array(tList)
    # Thresholding
    if THRESHOLDED:
        tList = (tList >0)*tList
    if SCALED:
        # Scale the dB mag spec to +1/-1
        maxVal = np.max(tList)
        minVal = np.min(tList)
        tList = (tList-minVal)/(maxVal-minVal)
    return np.array(tList)
'''


@keras.saving.register_keras_serializable()
def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(K.round(y_pred), 'int32')
    TP = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    TN = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    FP = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    FN = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



###########################################################################################



# Load CNN
model = tf.keras.models.load_model(CNN_MODEL_PATH, custom_objects={'f1_metric': f1_metric})

sensitivities = []
FARs = []
filenames = os.listdir("C:\\Users\\Amogh\\OneDrive - University of Cambridge\\Programming-New\\CaracalChitalDetector\\cnn\\data")
C_calls_all = []
model = tf.keras.models.load_model(CNN_MODEL_PATH)

for filename_count, filename in enumerate(filenames):
    
    ONE_HOUR_FILE_PATH = "C:\\Users\\Amogh\\OneDrive - University of Cambridge\\Programming-New\\CaracalChitalDetector\\cnn\\data\\" + filename
    SELECTION_TABLE_FILE_PATH = ONE_HOUR_FILE_PATH[:-3] + "Table.1.selections.txt"
    SELECTION_TABLE_FILE_PATH = SELECTION_TABLE_FILE_PATH.replace("data", "annotations")
    
    # 1 hour file - load anything you like 
    data, sr = librosa.load(ONE_HOUR_FILE_PATH, sr=8000)

    total_iterations = len(data)//int(1.5*sr)
    printProgressBar(0, total_iterations, prefix = f'Progress {filename_count+1}/{len(filenames)} {1}/{total_iterations} {filename}:', suffix = 'Complete', length = 50)

    C_calls = []
    
    # Iterate through each 1.5 second (half a window)
    for i, LB in enumerate(range(0, len(data)-int(1.5*sr), int(1.5*sr))):
    
        # Get lB
        UB = LB + (3*sr)
        
        # Extract clip
        clip = data[LB:UB]

        # If clip is empty, skip
        if len(clip) < sr*3.0:
            continue
        
        # Get spectral image using special function
        clipImg = wavFileToSpecImg(clip,num_strides=ML_NUM_FFT_STRIDES)
        
        # Reshape to (num_samples, height, width, channels) for use of CNN
        clipImg = clipImg.reshape(1, clipImg.shape[0], clipImg.shape[1], 1)
        
        # Run through CNN and get prediction(
        result = model.predict(clipImg, verbose=0)
        
        # If positive prediction and passes threshold, add to list
        if np.argmax(result[0]) == 1 and result[0][1] > 0.5:
            C_calls.append((LB/sr, UB/sr))

        printProgressBar(i + 1, total_iterations, prefix = f'Progress {filename_count+1}/{len(filenames)} {i+1}/{total_iterations} {filename}:', suffix='Complete', length=50)

    with open(ANNOTATION_FILE,'rb') as f:
        df = pickle.load(f)

    # Filter by positive (C) calls
    df = df[df['Annotation'] == 'C']
    
    # Filter by the selection table of the 1 hour file you are looking at
    df = df[df['SourceFile'] == SELECTION_TABLE_FILE_PATH]
    
    # Filter unused columns
    df = df.drop(columns=['SourceFile', 'LocationName', 'AnnotationType', 'StartTime', 'EndTime', 'FileStartTime', 'LowFreq', 'HighFreq', 'Annotation'])
    
    TP = 0
    FN = 0
    FP = 0
    totaltime = len(data) / sr / 3600
    
    # Go through each actual annotation and try and find a matching detection by the CNN
    # If found, incriment TP 
    # If went through all annotations and didn't find, increment FN
    # <start/end> means true call
    # <start/end>1 means detected call
    for start, end in zip(df['RelativeStartTime'], df['RelativeEndTime']):
        start = start.to_pytimedelta().total_seconds()
        end = end.to_pytimedelta().total_seconds()
        found = False
        for start1, end1 in C_calls:
            if (start1 >= start and end1 <= end) or (start >= start1 and end <= end1) or (start1 >= start and end1 >= end and start1 <= end) or (start >= start1 and end >= end1 and start <= end1):
                TP += 1
                found = True
                break
        if not found:
            FN += 1
    
    # Go through each detection and see if an annotation exists, if not, then it is a FP
    for start, end in C_calls:
        found = False
        for start1, end1 in zip(df['RelativeStartTime'], df['RelativeEndTime']):
            start1 = start1.to_pytimedelta().total_seconds()
            end1 = end1.to_pytimedelta().total_seconds()
            if (start1 >= start and end1 <= end) or (start >= start1 and end <= end1) or (start1 >= start and end1 >= end and start1 <= end) or (start >= start1 and end >= end1 and start <= end1):
                found = True
                break 
        if not found:
            FP += 1
            C_calls_all.append((filename, start, end))

    print()
    
    if TP+FN == 0:
        sensitivities.append(None)
        print(None)
    else:
        sensitivities.append(FN/(TP+FN))
        print(FN/(TP+FN))
    FARs.append(FP/totaltime)
    print(FP/totaltime)
    print()

print(sensitivities)
print(FARs)


###########################################################################################


# OLD MODEL: (model.keras)
x_old = [0.75, 0.0, None, None, None, None, None, None, 0.006211180124223602, None, None, None, 0.05405405405405406, None, None, 0.09375, 0.15428571428571428, None, None, None, 0.5, None, 0.2916666666666667, 0.16666666666666666, 0.021739130434782608, 0.03260869565217391]
y_old = [5.21557719054242, 65.81513260530421, 17.866866321807787, 265.9716296928328, 43.00458715596331, 18.001920204821847, 18.997973549488055, 38.995840443686006, 146.0155749946661, 0.9998933447098975, 29.00309366332409, 10.998826791808874, 32.99648037542662, 21.997653583617748, 85.00906763388095, 41.004373799871985, 30.996693686006825, 9.456960322797578, 0.0, 10.001066780456583, 25.997226962457336, 27.002880307232772, 42.004480477917646, 12.525979809976247, 125.9865614334471, 87.00928098997227]

# NEW MODEL: 50/50 old
x_new = [0.375, 0.13333333333333333, None, None, None, None, None, None, 0.049689440993788817, None, None, None, 0.02702702702702703, None, None, 0.17708333333333334, 0.21714285714285714, None, None, None, 1.0, None, 0.375, 0.0, 0.021739130434782608, 0.07065217391304347]
y_new = [79.5375521557719, 36.56396255850234, 65.08644160087123, 56.99392064846416, 66.00704075101345, 88.00938766801792, 163.9825085324232, 141.98485494880546, 76.00810753147003, 146.98432167235495, 38.004053765735016, 44.99520051194539, 58.993707337883954, 143.98464163822524, 66.00704075101345, 26.002773629187114, 28.99690699658703, 25.21856086079354, 58.00618732664818, 110.01173458502241, 37.99594709897611, 39.00416044378067, 107.01141455088543, 36.186163895486935, 50.99456058020478, 61.00650736078515]


# NEW MODEL 2: weights old
x_new_2 = [0.25, 0.13333333333333333, None, None, None, None, None, None, 0.043478260869565216, None, None, None, 0.0, None, None, 0.13541666666666666, 0.2057142857142857, None, None, None, 1.0, None, 0.375, 0.25, 0.09420289855072464, 0.08152173913043478]
y_new_2 = [100.39986091794158, 30.713728549141965, 61.257827389055265, 43.995307167235495, 50.00533390228291, 78.00832088756134, 222.97621587030716, 171.9816552901024, 68.00725410710476, 150.98389505119454, 30.003200341369748, 39.9957337883959, 54.99413395904437, 114.98773464163823, 53.00565393641989, 17.00181352677619, 25.997226962457336, 28.370880968392733, 60.006400682739496, 96.0102410923832, 33.99637372013652, 30.003200341369748, 95.01013441433753, 33.402612826603324, 34.996267064846414, 51.00544058032857]

# NEW MODEL 3: weights new
x_new_3 = [0.625, 0.0, None, None, None, None, None, None, 0.018633540372670808, None, None, None, 0.10810810810810811, None, None, 0.1875, 0.14857142857142858, None, None, None, 0.5, None, 0.375, 0.25, 0.021739130434782608, 0.07065217391304347]
y_new_3 = [3.911682892906815, 58.50234009360375, 22.971685270895726, 248.9734428327645, 40.00426712182633, 24.0025602730958, 15.99829351535836, 48.994773890784984, 138.01472157030085, 0.9998933447098975, 27.002880307232772, 7.99914675767918, 31.99658703071672, 25.997226962457336, 90.00960102410924, 41.004373799871985, 33.99637372013652, 3.1523201075991927, 1.0001066780456582, 6.00064006827395, 28.99690699658703, 27.002880307232772, 37.003947087689355, 15.309530878859858, 122.9868813993174, 85.00906763388095]

# NEW MODEL 3: 50/50 new
#bad

# NEW MODEL 4: weights new with lr=0.0001
x_new_4 = [0.625, 0.13333333333333333, None, None, None, None, None, None, 0.006211180124223602, None, None, None, 0.0, None, None, 0.21875, 0.13714285714285715, None, None, None, 0.5, None, 0.4583333333333333, 0.0, 0.043478260869565216, 0.04891304347826087]
y_new_4 = [1.303894297635605, 17.550702028081123, 11.485842635447863, 90.99029436860069, 23.002453595050138, 1.0001066780456582, 3.99957337883959, 26.997120307167236, 43.00458715596331, 0.0, 4.000426712182633, 4.999466723549488, 9.998933447098976, 22.997546928327644, 32.00341369746106, 12.0012801365479, 10.998826791808874, 0.0, 3.000320034136975, 6.00064006827395, 18.997973549488055, 23.002453595050138, 29.00309366332409, 22.268408551068884, 25.997226962457336, 33.003520375506724]


# Manually remove the None values when calcualating mean
print(np.mean([x for x in x_old if x != None]))
print(np.mean([y for y in y_old if y != None]))
#print()
#print(np.mean([x for x in x_new if x != None]))
#print(np.mean([y for y in y_new if y != None]))
#print()
#print(np.mean([x for x in x_new_2 if x != None]))
#print(np.mean([y for y in y_new_2 if y != None]))
print()
print(np.mean([x for x in x_new_3 if x != None]))
print(np.mean([y for y in y_new_3 if y != None]))
print()
print(np.mean([x for x in x_new_4 if x != None]))
print(np.mean([y for y in y_new_4 if y != None]))