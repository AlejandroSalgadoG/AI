import random

from Parameters import *

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

    dividedData = [ line.split(",") for line in linedData]

    return dividedData

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
