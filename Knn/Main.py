#!/bin/python

from Parameters import *
from Gui import *
from DataHandler import *
from Knn import *

def main():
    gui = Gui()

    rawData = readDataset(dataPath)
    data = parseData(rawData)

    gui.drawData(data)

    training, heldout, test = divideData(data)

    rawdata = unlableData(heldout)

    for sample in heldout:
        classification = classify(training, sample)
        print(sample, classification)

    gui.getMouse()

main()
