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

    p1 = [2, 3]
    p2 = [3, 4]

    weight = [p1,p2]

    gui.drawDivision(weight)

    gui.getMouse()

#    weight = learn(training)

main()
