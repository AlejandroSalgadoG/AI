import random

from Parameters import *
from Structures import *

ghost = None

def rand():
    return random.randint(0,100) / 100

def generateGhost():
    global ghost
    i = random.randint(0,2)
    j = random.randint(0,2)
    ghost = (i,j)

def getInitialPosDist():
    posNum = numRow**2
    prob = 1 / posNum

    return [prob for i in range(posNum)]

def getNewPosDist(pos, color, probs):
    dist = 
    return [0.11 for i in range(9)]

def calcDistMatrix(ghostPos):
    distMatrix = []
    for i in range(numRow):
        for j in range(numRow):
            dist = getGhostDistance(i,j, ghostPos)
            distMatrix.append(dist)

    return distMatrix

def useSensor(pos):
    dist = getGhostDistance(pos)
    color = selectRandom(model[dist])
    return translation[color]

def getGhostDistance(pos):
    i,j = pos
    gi, gj = ghost
    return abs(i-gi) + abs(j-gj)

def selectRandom(probs):
    randVal = rand()
    size = len(probs)

    for i in range(size):
        randVal -= probs[i]
        if randVal <= 0:
            return i

    return size-1
