#!/usr/bin/env python
# coding: utf-8

# # Template Runner 003
# This is a simple spectral masking (cross-correlation) based approach. We extract and generate a template, and then cross-correlate with peak detection. We export the template and the parameters used to a .h (header file) for export on the CARACAL ARM M4F board.
# 
# We run the template

# In[1]:


import librosa
#import pylab
import numpy as np
import scipy

import glob
import sys
import datetime


# In[ ]:


FILEPREFIX = "templateMaker003_001" 


# # Step 0: Load the template (generated earlier)

# In[ ]:


import pickle
with open(FILEPREFIX+".pkl",'rb') as f:
    templateObj = pickle.load(f)
tList = templateObj['tList']
DATECODE = templateObj["DateCode"]
ML_SR = templateObj['ML_SR']
SPECD_FFT_LEN = templateObj['SPECD_FFT_LEN']
ML_BIN_AGG = templateObj['ML_BIN_AGG']
ML_FLO = templateObj['ML_FLO']
ML_FHI = templateObj['ML_FHI']
ML_FFT_STRIDE = templateObj['ML_FFT_STRIDE']
WIN_LENGTH =  templateObj['WIN_LENGTH']
WIN_ALPHA = templateObj['WIN_ALPHA']
INIT_WINDEV = templateObj["INIT_WINDEV"]
WIN_ALPHA_MEAN = templateObj['WIN_ALPHA_MEAN']


def get_info(path):
    filename = path[110:]
            
    # Getting deviceno from path as it can be variable length (but not storing it)
    i = 3
    while filename[i] != '_':
        i += 1
    
    year = int(filename[i+1:i+5])
    
    return filename, year

# In[ ]:

files = glob.glob("C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/Test set/1 hour files/*.wav",recursive=True)
#files=glob.glob(sys.argv[1],recursive=True)

fout = open('C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/autodetect.txt','wt')
#fout=open(sys.argv[2],'wt')
    
fnk=0
for fn in files:
    # Gets the filename (without path) and the year and will ignore any 2023 files as they are badly labelled
    filename_chopped, year = get_info(fn)
    if year < 2024:
        continue

    fnk=fnk+1
    print('{}\t{}\n'.format(fnk, filename_chopped))
    
    filename = fn
    #filename = "C:\\CloudData\\2024\\Nepal\\ML002\\file_1711184400.wav"
    
    fout.write('{}\t'.format(filename_chopped))
    
        
    aud,sr = librosa.load(filename,sr=ML_SR)
        
    startT = 0 # Time in seconds to extract a useful clip from
    endT = int(len(aud)/sr)
    #gtT = [44,45.7,46.6, 48.9,49.55,51.8,52.1,54.0,54.7,58.4,61.0,62.4,66.9,67.5,69.3,71.2,72.1,72.4] # Ground Truth call times
    
            
    #print(f"File Samples:{np.shape(aud)}, Rate:{sr}")
    aud = aud[startT*sr:endT*sr]
    #print(f"Clipped Samples:{np.shape(aud)}, Rate:{sr}")
    
    
    # # Step 1:
    # 
    # Convert the wav file to FFT. We then extract out our "feature map" which is just the spectral magnitude bins. We do a simple boxcar aggregate, but we could use a triangular weighting quite easily as well.
    
    # In[ ]:
    
    
    def chunkToBins(chunk,fLo,fHi,numbins,sr):
        """convert a chunk (window) to slope.
        Provide the low and high frequencies in Hz for a spectral windowing
        numbins are the number of output bins between flo and fhi
        Provide the sample rate in Hz"""
        CMPLX_FFT_LEN = len(chunk)*2
        
        fS = np.fft.fft(chunk,n=CMPLX_FFT_LEN) # fft - note we double it for the cmplx fft
        fRes = sr/(CMPLX_FFT_LEN)   # frequency per cmplx bin
        #print(fRes)
        binLo = int(fLo/sr*CMPLX_FFT_LEN)
        binHi = int(fHi/sr*CMPLX_FFT_LEN)
        specSize = int((binHi-binLo)/numbins)
        binTotals = np.zeros(numbins)
        for k in range(numbins):
            dbSum = 0
            for j in range(specSize):
                idx = binLo + (k * numbins) + j
                dbVal = np.log10(np.abs(fS[idx]))
                dbSum += dbVal
            binTotals[k] = dbSum
        return binTotals
    q = chunkToBins(aud[:SPECD_FFT_LEN],ML_FLO,ML_FHI,ML_BIN_AGG,ML_SR)
    #print(q)
    
    
    
    # # Step 2: Show the template
    
    # In[ ]:
    
    
    #print(np.min(tList),np.max(tList))
    
    #pylab.imshow(tList.T,aspect=10)
    #pylab.show()
    
    
    # # Step 3: Slide the window and correlate
    # 
    # We manually compute the correlation, so it is directly the same as our high-tech C code.
    
    # In[ ]:
    
    
    qList = []
    for idx in range(0,len(aud),int(ML_FFT_STRIDE)):
        clip = aud[idx:idx+SPECD_FFT_LEN]
        q = chunkToBins(clip,ML_FLO,ML_FHI,ML_BIN_AGG,ML_SR)
        
        qList.append(q)
    qList = np.array(qList)
    # Now we cross correlate
    #print(np.shape(qList),np.shape(tList))
    xcorr = []
    for offset in range(len(qList)-np.shape(tList)[0]):
        xcTotal = 0
        for tIdx in range(np.shape(tList)[0]):
            for bIdx in range(ML_BIN_AGG):
                xcTotal += qList[offset+tIdx][bIdx]*tList[tIdx][bIdx]
        xcorr.append(xcTotal)
    xcorr = np.array(xcorr)
    
    
    #print(np.shape(xcorr))
    
    
    # pylab.subplot(311)
    # pylab.imshow(qList.T,aspect=50)
    # pylab.xlim(0,len(qList))
    # pylab.subplot(312)
    # pylab.specgram(aud,Fs=8000,NFFT=1024,noverlap=800)
    # pylab.ylim(800,1400)
    # pylab.scatter(gtT,np.ones(len(gtT))*1000,c='red',marker='*',s=80)
    # pylab.subplot(313)
    # pylab.plot((xcorr))
    # pylab.xlim(0,len(xcorr))
    # pylab.show()
    
    
    # # Step 4: Peak Detection
    # 
    # This is loosely based on https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/54507140#54507140
    # 
    # But made simpler to work neatly on the MCU.
    
    # In[ ]:
    
    
    WIN_HOP =1
    WIN_THRESHOLD = 10
    #WIN_LENGTH = 15
    
    dets = []
    scaled_vals = []
    devs = []
    means = []
    for k in range(WIN_LENGTH):
            dets.append(0)
            scaled_vals.append(0)
    
    winDev=INIT_WINDEV
    alpha = WIN_ALPHA
    winMean = np.mean(xcorr[:WIN_LENGTH])
    alpha_mean = WIN_ALPHA_MEAN
    for idx in range(0,len(xcorr)-WIN_LENGTH,WIN_HOP):
        # This is our circular window
        extract = np.array(xcorr[idx:idx+WIN_LENGTH])
        winDev = (alpha*np.std(extract)) + (1-alpha)*winDev
        devs.append(winDev)
        winMean = (alpha_mean*np.mean(extract))+(1-alpha_mean)*winMean
        means.append(winMean)
        det = 0
        if (extract[-1] -winMean)/winDev > WIN_THRESHOLD:
            det = 1
            fout.write('{}\t'.format(idx*ML_FFT_STRIDE/sr))
        scaled_vals.append((extract[-1] -winMean)/winDev)
        dets.append(det)
    # pylab.subplot(511)
    # pylab.plot(xcorr)
    # pylab.subplot(512)
    # pylab.plot(scaled_vals)
    # pylab.subplot(513)
    # pylab.plot(means,'r')
    # pylab.subplot(514)
    # pylab.plot(devs,'g')
    # pylab.subplot(515)
    # pylab.plot(dets)
    
    # pylab.show()
    
    fout.write('\n')
        
fout.close()