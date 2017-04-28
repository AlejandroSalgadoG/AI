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

    for classdata in classData:
        for feature in range(featureNum):
            probs = mlSmooth(classdata, feature)
            print(probs)

main()
