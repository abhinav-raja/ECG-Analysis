import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


def findPeaksViaDispersion(signal, lag, stdDevThreshold):
    """Returns a binary signal representing """

    #Initialise variables
    result = np.zeros(len(signal))
    filteredY = signal[:lag]
    filteredY = np.append(filteredY, np.zeros(len(signal) - lag))
    avg = np.mean(filteredY[:lag])
    stdDev = np.std(filteredY[:lag])

    for i in range(lag, len(signal)):
        if signal[i] - avg > stdDevThreshold*stdDev:
            result[i] = 1
            filteredY[i] = avg
        else:
            result[i] = 0
            filteredY[i] = signal[i]
        avg = np.mean(filteredY[i-lag+1 : i + 1])
        stdDev = np.std(filteredY[i-lag+1 : i + 1])
    return result

def detectQRS(ecgLeadData, samplingFreq):
    """docstring goes here"""
    #Bandpass filter frequencies - 5 and 15 Hz normalised by sampling rate
    w1 = 10/samplingFreq
    w2 = 30/samplingFreq 
    b, a = sig.butter(4, [w1, w2], 'bandpass')
    bandpassedSignal = sig.filtfilt(b, a, ecgLeadData)
    differentiatedSignal = np.diff(bandpassedSignal)
    squaredSignal = np.power(differentiatedSignal, 2)

    #Window should cover ~100-150ms
    windowLength = int(0.15*samplingFreq)
    movingWindowAverage = np.ones(windowLength)/windowLength
    #Note that this convolution induces a delay of windowLength/2 in the result
    averagedSignal = np.convolve(squaredSignal, movingWindowAverage)

    #lag should be ~300ms for best results
    peakSignal = findPeaksViaDispersion(averagedSignal, int(0.3*samplingFreq), 5)
    edges = np.diff(peakSignal)

    #Store the onset and offset of each qrs
    qrsLocationSamples = np.argwhere(peakSignal)
    if peakSignal[0] = 1