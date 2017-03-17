import sys
import random

def getPositionMatrix():
    return [[0.11 for j in range(9)] for i in range(9)]

def getColorMatrix():
    return [['E' for j in range(9)] for i in range(9)]

def printMatrix(matrix, prob):
    print
    for i in range(3):
        for j in range(3):
            if prob:
                sys.stdout.write( "%.2f " % matrix[i][j] )
            else:
                sys.stdout.write( matrix[i][j] + " " )
        print

def getRandPos(cmatrix):
    empty = getEmptySpot(cmatrix)

    size = len(empty)
    if size == 0:
        return None

    pos = random.randint(0,size-1)
    return empty[pos]

def getEmptySpot(cmatrix):
    empty = []

    for i in range(3):
        for j in range(3):
            if cmatrix[i][j] == 'E':
                empty.append((i,j))
    return empty
