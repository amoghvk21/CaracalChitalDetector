#!/usr/bin/env python
# coding: utf-8

# # Running CNN
# Given the model and a 1 hour file, output the locations of the calls in the file

# ### Imports

# In[1]:


import tensorflow as tf
import numpy as np
import librosa
import pickle
import time


# ### Parameters

# In[2]:


ML_SR = 8000 # Target sampling rate
SPECD_FFT_LEN =  512 # Real FFT length (in the M4F - we use double of this on the PC as we don't do single-sided)
ML_BIN_AGG = 14 # Number of frequency bins (vertical dimension)
ML_FLO = 600 # Low freq
ML_FHI = 1400 # High freq
ML_FFT_STRIDE=1024 # Stride length in audio samples
ML_NUM_FFT_STRIDES = 12 # How many strides make up a sample
THRESHOLDED = False # Threshold the template or not

ONE_HOUR_FILE_PATH = "C:\\Users\\Amogh\\OneDrive - University of Cambridge\\Programming-New\\CaracalChitalDetector\\cnn\\data\\CAR204_20240323$160400_1711189140.wav"
#ONE_HOUR_FILE_PATH = "C:\\Users\\Amogh\\OneDrive - University of Cambridge\\Programming-New\\CaracalChitalDetector\\cnn\\output\\POS_000001.wav"
CNN_MODEL_PATH = "C:\\Users\\Amogh\\OneDrive - University of Cambridge\\Programming-New\\CaracalChitalDetector\\cnn\\03MLouput20240811v001.keras"
CNN_OUTPUT_FILE_PATH = "output.pkl"


# ### "Simple magic" functions
# Takes a clip and returns an spectral image 

# In[3]:


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


# In[4]:


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


# ### Load CNN and random 1 hour file 

# In[5]:


# Load CNN
model = tf.keras.models.load_model(CNN_MODEL_PATH)

# 1 hour file - load anything you like 
data, sr = librosa.load(ONE_HOUR_FILE_PATH, sr=8000)

print(data)
print(len(data)-int(3*sr))


# ### Main loop
# Iterate through each clip in 1 hour file, run special function and run CNN \
# Iterating each 1.5 second which is half a window \
# Results in half an overlap

# In[6]:


C_calls = []

print(f'Total iterations required: {(len(data)//int(1.5*sr))}')

start = time.time()

# Iterate through each 1.5 second (half a window)
for i, LB in enumerate(range(0, len(data)-int(1.5*sr), int(1.5*sr))):

    # Get lB
    UB = LB + (3*sr)
    
    # Extract clip
    clip = data[LB:UB]
    
    # Get spectral image
    clipImg = wavFileToSpecImg(clip,num_strides=ML_NUM_FFT_STRIDES)
    
    # Reshape to (num_samples, height, width, channels)
    clipImg = clipImg.reshape(1, clipImg.shape[0], clipImg.shape[1], 1)
    
    # Run model and get result
    result = model.predict(clipImg, verbose=0)
    
    # Append to total C calls if necessary
    if np.argmax(result[0]) == 1:
        C_calls.append((LB/sr, UB/sr))
    
    print(i+1)
    
end = time.time()

print(C_calls)
print(f'total time taken: {end-start}')


# ### Save results

# In[ ]:


with open(CNN_OUTPUT_FILE_PATH, 'wb') as f:
    pickle.dump(C_calls, f)


# In[7]:


print(len(C_calls))


# ### Test code

# In[10]:


import glob
for filepath in glob.glob(r"C:\Users\Amogh\OneDrive - University of Cambridge\Programming-New\CaracalChitalDetector\cnn\output\*.wav"):
    if 'POS' in filepath:
        
        # Extract clip
        clip, sr = librosa.load(filepath, sr=8000)
        
        # Get spectral image
        clipImg = wavFileToSpecImg(clip, num_strides=ML_NUM_FFT_STRIDES)
        
        # Reshape to (num_samples, height, width, channels)
        clipImg = clipImg.reshape(1, clipImg.shape[0], clipImg.shape[1], 1)
        
        # Run model and get result
        result = model.predict(clipImg, verbose=0)
        
        print(np.argmax(result[0]))

