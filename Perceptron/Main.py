#!/bin/python

from Parameters import *
from DataHandler import *

def main():
    rawData = readDataset(dataPath)
    data = parseData(rawData)

    training, heldout, test = divideData(data)

main()
