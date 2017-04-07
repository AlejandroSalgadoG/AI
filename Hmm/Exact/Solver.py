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
    dist = getGhostDistance(pos, ghost)
    color = selectRandom(model[dist])
    return translation[color]

def getGhostDistance(pos, ghost):
    i,j = pos
    gi, gj = ghost
    return abs(i-gi) + abs(j-gj)

def getNewDist(pos, color, probs):
    newDist = []

    for i in range(numRow):
        for j in range(numRow):
            imgGhost = (i,j)

            pex = calcCondProb(pos, color, imgGhost)
            px = getPosProb(imgGhost, probs)

            newDist.append(pex * px)

    return normalize(newDist)

def calcCondProb(pos, color, imgGhost):
    dist = getGhostDistance(pos, imgGhost)

    idx = translation[color]
    return model[dist][idx]

def getPosProb(pos, probs):
    i,j = pos
    return probs[j+i*numRow]

def normalize(dist):
    total = 0
    size = len(dist)

    for i in range(size):
        total += dist[i]

    for i in range(size):
        dist[i] /= total

    return dist

def selectRandom(probs):
    randVal = rand()
    size = len(probs)

    for i in range(size):
        randVal -= probs[i]
        if randVal <= 0:
            return i

    return size-1

def moveGhost():
    global ghost
    table = transition[ghost]
    idx = selectRandom(table)
    ghost = fromIdxToPos(idx)

def fromIdxToPos(idx):
    i = idx // numRow
    j = idx - i * numRow
    return (i,j)
