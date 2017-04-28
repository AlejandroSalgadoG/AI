#!/bin/python

from Parameters import *
from DataHandler import *
from Solver import *

def main():
    rawData = readDataset(dataPath)
    data = parseData(rawData)
    treeTable = buildTable(data)

    #training, heldout, test = divideData(data)

    tree = desitionTree(treeTable)

main()
