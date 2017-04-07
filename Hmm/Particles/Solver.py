import random

from Parameters import *
from Model import *

ghost = None

def rand():
    return random.randint(0,100) / 100

def getInitialDist():
    posNum = numRow**2
    prob = 1 / posNum

    return [prob for i in range(posNum)]

def generateGhost():
    global ghost
    i = random.randint(0,2)
    j = random.randint(0,2)
    ghost = (i,j)

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

def getNewDist(pos, color, probs):
    return [0.11 for i in range(9)]
