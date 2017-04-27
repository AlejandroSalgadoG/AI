#!/bin/python

from Parameters import *
from Gui import *
from DataHandler import *
from Perceptron import *

def main():
    rawData = readDataset(dataPath)
    data = parseData(rawData)

    #training, heldout, test = divideData(data)

    alphas = learn(data)

main()
