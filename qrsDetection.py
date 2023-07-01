import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from typing import TypedDict

qrsOutputInfo = TypedDict('qrsOutputInfo', {'samples': np.ndarray, 'times': np.ndarray})

def findPeaksViaDispersion(signal: np.ndarray, lag: int, stdDevThreshold: float, minThreshold: float) -> np.ndarray :
    """
    Returns a binary signal representing peak regions in the input signal

    Arguments:

    `signal` - the signal in which to detect peaks

    `lag` - the size of the window in which mean and standard deviation are calculated

    `stdDevThreshold` - if the signal varies this many standard deviations from the mean, it's detected as a peak

    `minThreshold` - the minimum value the threshold can take
    """

    #Initialise variables
    result = np.zeros(len(signal))
    filteredY = np.append(signal[:lag], np.zeros(len(signal) - lag))
    avg = np.mean(filteredY[:lag])
    stdDev = np.std(filteredY[:lag])

    #Process signal
    for i in range(lag, len(signal)):
        if signal[i] - avg > max(stdDevThreshold * stdDev, minThreshold):
            result[i] = 1
            #Set filteredY to avg to minimise effect on standard deviation
            filteredY[i] = avg
        else:
            result[i] = 0
            filteredY[i] = signal[i]
        avg = np.mean(filteredY[i - lag + 1 : i + 1])
        stdDev = np.std(filteredY[i - lag + 1 : i + 1])
    return result

def detectQRS(ecgLeadData: np.ndarray, samplingFreq: int) -> qrsOutputInfo :
    """
    Returns a list of pairs representing QRS starting and ending regions

    Arguments:

    `ecgLeadData`: data from any single lead of an ECG

    `samplingFreq`: sampling frequency of the data

    Output keys:

    `dict.samples`: start and end samples for each QRS interval

    `dict.times`: start and end times for each QRS interval
    """
    #Bandpass filter frequencies - 5 and 15 Hz normalised by sampling rate
    w1 = 10/samplingFreq
    w2 = 30/samplingFreq 
    b, a = sig.butter(4, [w1, w2], 'bandpass')

    #Preliminary signal processing
    bandpassedSignal = sig.filtfilt(b, a, ecgLeadData)
    differentiatedSignal = np.diff(bandpassedSignal)
    squaredSignal = np.power(differentiatedSignal, 2)

    #Moving window average: window should voer ~100-150ms
    windowLength = int(0.15*samplingFreq)
    movingWindowAverage = np.ones(windowLength)/windowLength
    #Note that this convolution induces a delay of windowLength/2 in the result
    averagedSignal = np.convolve(squaredSignal, movingWindowAverage)

    #lag should be ~300ms for best results
    peakSignal = findPeaksViaDispersion(averagedSignal, int(0.3*samplingFreq), 3, 2*np.mean(averagedSignal))
    edges = np.diff(peakSignal)

    #Store the onset and offset of each qrs
    qrsLocationSamples = np.argwhere(edges).flatten()
    #Remove incomplete QRS regions in the beginning/end of the signal
    if edges[qrsLocationSamples[0]] == -1:
        qrsLocationSamples = qrsLocationSamples[1:]
    if edges[qrsLocationSamples[-1]] == 1:
        qrsLocationSamples = qrsLocationSamples[:-1]
    
    #Prepare and format the output
    qrsLocationSamples = np.reshape(qrsLocationSamples, (-1, 2))
    qrsLocationTimes = np.array([[el/samplingFreq for el in interval] for interval in qrsLocationSamples])
    qrsLocations = {
        "samples": qrsLocationSamples,
        "times": qrsLocationTimes
    }

    return qrsLocations