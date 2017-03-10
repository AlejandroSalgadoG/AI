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

visited = [[0,0],[0,0]]
probAc = 0
def updateProbMatrix(prob, pmatrix, x, y):
    global probAc

    prob *= 0.25
    count = 0
    comp = 1 - prob - probAc
    pmatrix[x][y] = prob
    for i in range(2):
        for j in range(2):
            if i == x and j == y:
                visited[i][j] = 1
                probAc += prob
            else:
                if visited[i][j] == 0:
                    count += 1
    if count != 0:
        prob = comp / count

    for i in range(2):
        for j in range(2):
            if visited[i][j] == 0:
                pmatrix[i][j] = prob

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

        updateProbMatrix(prob, pmatrix, x, y)

main()
