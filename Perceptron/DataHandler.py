import random

from Parameters import *
from Standars import *

def rand(lower, upper):
    return random.randint(lower, upper) 

def readDataset(path):
    dataFile = open(path, "r")
    data = dataFile.read()
    dataFile.close()

    return data

def parseData(rawData):
    linedData = rawData.split("\n")
    del linedData[-1]

    parsedData = []

    for line in linedData:
        tmpData = line.split(",")
        classification = tmpData[-1] # get classification
        del tmpData[-1]

#        data = [1] # bias
        data = [] 
        for element in tmpData:
            data.append(float(element)) # get features values

#        data.append(classes[classification]) # get code of the classification
        data.append(classification) # get code of the classification

        parsedData.append(data) # store the sample

    return parsedData

def divideData(data):
    size = len(data)

    testSz = int(size * testRate)
    heldSz = int(size * heldoutRate)
    trainSz = size - (testSz + heldSz)

    train = []
    heldout = []
    test = []

    for i in range(size-1, -1, -1):
        pos = rand(0, i)
        sample = data[pos]

        if size-i <= trainSz:
            train.append(sample)
        elif size-i <= heldSz+trainSz:
            heldout.append(sample)
        else:
            test.append(sample)

        del data[pos]

    return train, heldout, test
