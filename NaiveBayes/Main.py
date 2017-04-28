#!/bin/python

from Parameters import *
from DataHandler import *
from Solver import *

def main():
    rawData = readDataset(dataPath)
    data = parseData(rawData)

    #training, heldout, test = divideData(data)

    classData = []

    for key in classes:
        classId = classes[key]
        classdata = getConsistentData(classId, data)
        classdata = removeLabel(classdata)
        classData.append(classdata)

    featureNum = len(classData[0][0])

    condProb = []
    for classdata in classData:
        classProb = []
        for feature in range(featureNum):
            probs = mlSmooth(classdata, feature)
            classProb.append(probs)
        condProb.append(classProb)

    sample = [1,2]

    classification = naiveBayes(condProb, sample)

    print(translate[classification])

main()
