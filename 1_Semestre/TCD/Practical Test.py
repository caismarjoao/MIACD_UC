# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:02:24 2024

@author: Marta Antunes
"""

import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.getcwd(), "dataset")

fileDict = {}
keys = np.zeros(7)


def extractAudio():
    global fileDict
    
    # Define paths for each class of audio data
    files_fo = os.path.join(path, "forward")
    filesFo = os.listdir(files_fo)
    
    files_ba = os.path.join(path, "backward")
    filesBa = os.listdir(files_ba)
    
    files_le = os.path.join(path, "left")
    filesLe = os.listdir(files_le)
    
    files_ri = os.path.join(path, "right")
    filesRi = os.listdir(files_ri)
    
    files_st = os.path.join(path, "stop")
    filesSt = os.listdir(files_st)
    
    files_un = os.path.join(path, "_unknown_")
    filesUn = os.listdir(files_un)
    
    files_si = os.path.join(path, "_silence_")
    filesSi = os.listdir(files_si)
  
    # Populate the dictionary with file names for each category
    fileDict = {
        "forward": filesFo,
        "backward": filesBa,
        "left": filesLe,
        "right": filesRi,
        "stop": filesSt,
        "_unknown_": filesUn,
        "_silence_": filesSi
    }


def normedSound(filename, key):
    fs, data = wavfile.read(path + "/" + key + "/" + filename)  # Read audio file
    
    dataType = data.dtype  # Get data type (e.g., int16)
    maxValue = np.iinfo(dataType).max  # Get the maximum possible value for the data type
    
    dataNormalized = data / maxValue  # Normalize audio data
    
    return dataNormalized, fs


def envelope(data, window):    
    absData = np.abs(data)  # Take the absolute value of the data
    envelopeData = np.zeros_like(absData)  # Initialize an array for the envelope
    
    halfWindow = window // 2  # Compute half the window size
    
    # Compute the envelope by averaging values within the window
    for i in range(halfWindow, len(envelopeData) - halfWindow):
        envelopeData[i] = np.mean(absData[i - halfWindow : i + halfWindow])
        
    return envelopeData


def downsampling(arrayAux):
    size = 0  # Counter for downsampled array size
    
    # Count downsampled array size by stepping through every 50th element
    for i in range(0, len(arrayAux), 50):        
        size += 1
        
    arrayDownsampled = np.zeros(size)  # Initialize downsampled array
    
    j = 0
    # Populate downsampled array by taking every 50th element
    for i in range(0, len(arrayAux), 50):
        arrayDownsampled[j] = arrayAux[i]
        j += 1
        
    return arrayDownsampled


def question1(key):    
    i = 0
    
    stds = []
    
    for item in fileDict[key]:
        if i == 100:
            break
        
        dN, fs = normedSound(item, key)
        arrayAux = envelope(dN, 200)
        
        arrayDownsampled = downsampling(arrayAux)
        
        std = np.std(arrayDownsampled)
        stds.append(std)
        
        i += 1
        
    return stds
        
        
def zScore(data):
    mean = np.mean(data)
    std = np.std(data)
    
    return mean, std


def question2(data):
    indexesOutliers = []
    
    mean, std = zScore(data)
    threshold = 3.5  # Define Z-score threshold for outliers
    
    for index, x in enumerate(data):
        z = (x - mean) / std
        
        if np.abs(z) > threshold:
            indexesOutliers.append(index)
    
    return indexesOutliers



def question3(allStds):
    plt.figure(figsize=(10, 6))  # Set figure size for scatter plot

    # Iterate through each class's values to plot
    for i, stds in enumerate(allStds):
        indexes = question2(stds)  # Identify outliers
        
        outlierList = []  # List for outlier value
        normalList = []  # List for non-outlier value
        
        # Classify each average as outlier or normal
        for index in range(len(stds)):
            if index in indexes:
                outlierList.append(stds[index])  # Append outliers
            else:
                normalList.append(stds[index])  # Append normal points
        
        # Set x-axis positions for normal and outlier points
        xNormal = [i] * len(normalList)
        xOutliers = [i] * len(outlierList)
        
        # Plot normal and outlier value with distinct colors
        plt.scatter(xNormal, normalList, label = keys[i], color = (31/255, 119/255, 180/255))
        plt.scatter(xOutliers, outlierList, label = keys[i], color = (255/255, 119/255, 0/255))

    plt.xticks(range(len(keys)), keys)
    plt.ylabel('Standard Deviation')
    plt.show()
    
    

if __name__ == "__main__":
    extractAudio()
    keys = np.array(list(fileDict.keys()))
    
    allStds = []
    for key in fileDict.keys():
        stds = question1(key)
        allStds.append(stds)
        
    question3(allStds)
    
    print('THE END!')