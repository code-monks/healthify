import numpy as np
import cv2
import time
#import matplotlib.pyplot as plt
# from drawnow import *
from scipy.signal import butter, lfilter
from scipy.signal import  savgol_filter,filtfilt,welch,butter

from scipy.fftpack import fft
import os
import imageio
import time
import matplotlib.pyplot as plt
from scipy import linalg as sla
from scipy import signal
from scipy import interpolate

fs = 30.0
fps = 30

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    #high = highcut / nyq
    b, a = butter(order, [low], btype='low')
    y = filtfilt(b, a, data)
    return y


def plotValues():
    plt.figure(1)
    y = butter_bandpass_filter(scolor, lowcut, highcut, fs, order=6)
    ytrunc=y[180:]
    strunc=stime[180:]
    plt.plot(strunc, ytrunc, color='#fc4f30')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Pixel Color')
    plt.title('Filtered green Channel Pixel Data')

def main(pathToVideo):
    interval=20.0
    HRs=[]

    w1=[]
    w2=[]
    w3=[]
    Cum_trace1=[]
    Cum_trace2=[]

    u1=[]
    u2=[]
    u3=[]
    trace=[]
    HRs=[]
    frame_num=-1
    SR=np.zeros([1,3])
    start_t=time.time()
    start=start_t

    window = 3 #For sliding window
    intialFlag=3 #For sliding window

    min_YCrCb = np.array([0,133,98],np.uint8)
    max_YCrCb = np.array([255,177,142],np.uint8)

    capture = cv2.VideoCapture(pathToVideo) #From webcamera
    if not capture.isOpened():
        raise RuntimeError('Error opening VideoCapture.')
    ##capture.set(3,640)
    ##capture.set(4,480)

    #(grabbed, frame) = capture.read()
    #snapshot = np.zeros(frame.shape, dtype=np.uint8)
    stime=np.array([])
    scolor=np.array([])
    face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') 
    i=0
    colors = {'red': [], 'green': [], 'blue': []}
    while True:
        newtime=time.clock()
        i=i+1

        #Read a frame
        (grabbed, frame) = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        #cv2.imshow('Video', frame)
        #cv2.imwrite('TestResol.jpg',frame)
        #NEED FACE-MASK IMPLEMENTATION HERE
        frame_num +=1
        #Skinmask implementation
        transcol=cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        mask = cv2.inRange(transcol,min_YCrCb,max_YCrCb)
        #cv2.imshow('mask',mask)
        res = cv2.bitwise_and(transcol,transcol, mask= mask)
        #cv2.imshow('res',res)
        transcol2=cv2.cvtColor(res, cv2.COLOR_YCR_CB2BGR)
        #cv2.imshow('rest',transcol2)
        face=transcol2

        #Chosing the Region of Interest
        roi=face.astype(np.float64)

        rgb_pixels=np.reshape(roi,(-1,3),order='F')
        rgb_corr=np.dot(rgb_pixels.T,rgb_pixels)
        rgb_corr=rgb_corr/len(rgb_pixels)

        w,v=sla.eigh(rgb_corr) # w-eigen values, v- eigen vector
        w1.append(w[2])
        w2.append(w[1])
        w3.append(w[0])
        v=v.T

        u1.append(v[2])
        u2.append(v[1])
        u3.append(v[0])

        ref=1
        if frame_num>ref:
            ref=frame_num-1

            t=frame_num
            rotation=[[np.dot(u1[t].T,u2[ref]),np.dot(u1[t].T,u3[ref])]]
            scale=np.sqrt([w1[t]/w2[ref],w1[t]/w3[ref]])
            sr=scale* rotation
            plane=np.dstack((u2[ref],u3[ref]))
            sr_project=np.dot(sr,plane[0].T)
            SR=np.dstack((SR,sr_project))
        curr_t=time.time()

        k=cv2.waitKey(15) & 0xff
        if k==27: #or (curr_t -start > 12 * interval +2):
            break

        if curr_t- start_t > interval:
            trace1=SR[0][0,:]
            trace2=SR[0][1,:]
            trace3=SR[0][2,:]

            #Combining Trace 1
            Cum_trace1.append(trace1)
            Cum_trace1=Cum_trace1[-window:]
            trace1=[]
            for t in Cum_trace1:
                trace1.extend(t)

            trace1=np.array(trace1)

            #Combining Trace 2
            Cum_trace2.append(trace2)
            Cum_trace2=Cum_trace2[-window:]
            trace2=[]
            for t in Cum_trace2:
                trace2.extend(t)

            trace2=np.array(trace2)
            sigma = np.std(trace1) / np.std(trace2)

            frame_rate=float(frame_num)/interval

            trace=trace1- sigma * trace2
            if intialFlag >1:
                intialFlag = intialFlag-1
                del w2[:]
                del w1[:]
                del w3[:]
                del u1[:]
                del u2[:]
                del u3[:]

                SR=np.zeros([1,3])

                start_t=curr_t
                frame_num=-1

                face_box=None

                continue;

            filtered_trace=butter_bandpass_filter(trace,0.33,2.0,frame_rate,order=5)
            f,psd=signal.welch(filtered_trace,frame_rate,noverlap=128,nperseg=256,nfft=1024)
            heartRate=60.0 * f[np.argmax(psd)]
            HRs.append(heartRate)
            HRs=HRs[-5:]
            heartRate_mdn=np.median(HRs)
            heartRate_mean=np.mean(HRs)

            del w2[:]
            del w1[:]
            del w3[:]
            del u1[:]
            del u2[:]
            del u3[:]

            SR=np.zeros([1,3])

            start_t=curr_t
            frame_num=-1

            face_box=None
            #interval=10

    cap.stop()
    cv2.destroyAllWindows()
    ans = {
        'instHeartRate' : heartRate,
        'mdnHeartRate' : heartRate_mdn,
        'meanHeartRate' : heartRate_mean
    }
    return ans