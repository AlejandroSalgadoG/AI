#!/bin/python2.7

import sys

from Model import *

def getProbMatrix():
    return [[0.25 for j in range(2)] for i in range(2)]

def getColorMatrix():
    return [['E' for j in range(2)] for i in range(2)]

def printColorMatrix(matrix):
    print

    for i in range(2):
        for j in range(2):
            sys.stdout.write( matrix[i][j] + " " )
        print 

    print

def printProbMatrix(matrix):
    for i in range(2):
        for j in range(2):
            sys.stdout.write( "%.2f " % matrix[i][j] )
        print 

    print

def main():
    pmatrix = getProbMatrix()
    cmatrix = getColorMatrix()

    while True:
        printColorMatrix(cmatrix)
        printProbMatrix(pmatrix)
    
        user_input = raw_input("Enter (x, y) or x to exit = ")
        if user_input == "x":
            break
    
        pos = user_input.split(" ")
        x, y = int(pos[0]), int(pos[1])

        prob, color = sensorModel(x,y)
    
        cmatrix[x][y] = color

        print prob, color

main()
