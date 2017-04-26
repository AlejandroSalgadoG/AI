#!/bin/python

from Parameters import *
from Gui import *
from DataHandler import *
from Perceptron import *

def main():
    gui = Gui()

    rawData = readDataset(dataPath)
    data = parseData(rawData)


    training, heldout, test = divideData(data)
    gui.drawData(training)

    gui.getMouse()

    weight = learn(training)

    gui.drawDivision(weight)

    gui.getMouse()

main()
