#!/bin/python2.7

from Model import *
from MatrixHandler import *

def main():
    initializeModel()

    pmatrix = getPositionMatrix()
    cmatrix = getColorMatrix()
    posDist = getInitialPosDist()

    printMatrix(pmatrix, prob=True)
    printMatrix(cmatrix, prob=False)

    pos = getRandPos(cmatrix)

    while pos is not None:
        i,j = pos
        color = getColor()
        cmatrix[i][j] = color

        posDist = updatePosDist(posDist, color)
        pmatrix = updatePosMat(pmatrix, posDist)

        raw_input()

        printMatrix(pmatrix, prob=True)
        printMatrix(cmatrix, prob=False)
        
        pos = getRandPos(cmatrix)

main()
