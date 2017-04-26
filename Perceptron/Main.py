#!/bin/python

from Parameters import *
from Gui import *
from DataHandler import *
from Perceptron import *

def main():
    gui = Gui()

    rawData = readDataset(dataPath)
    data = parseData(rawData)

    gui.drawData(data)

#    training, heldout, test = divideData(data)

    gui.getMouse()

    weights = learn(data)

    if numClass == 2:
        gui.drawDivision(weights[0])

    gui.getMouse()

main()
