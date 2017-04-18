#!/bin/python

from Parameters import *
from DataHandler import *
from Perceptron import *

def main():
    rawData = readDataset(dataPath)
    data = parseData(rawData)

    training, heldout, test = divideData(data)

    weight = learn(training)
    print(weight)

main()
