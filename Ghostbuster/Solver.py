import random

from Parameters import *
from Model import *

ghost = None

def rand():
    return random.randint(0,100) / 100

def generateGhost():
    global ghost
    size = numRow - 1
    i = random.randint(0,size)
    j = random.randint(0,size)
    ghost = (i,j)

def isGhostThere(pos):
    if pos == ghost:
        return True
    else:
        return False

def getInitialPosDist():
    posNum = numRow**2
    prob = 1 / posNum

    return [prob for i in range(posNum)]

def getNewPosDist(pos, color, probs):
    newDist = []

    for i in range(numRow):
        for j in range(numRow):
            imgGhost = (i,j)

            dist = getGhostDistance(pos, imgGhost)
            table = model[dist]
            colorIdx = translation[color]
            posIdx = j + i*numRow

            psf = table[colorIdx]
            pf = probs[posIdx]

            newDist.append( psf * pf )

    return normalize(newDist)

def normalize(probs):
    acum = 0
    size = len(probs)

    for i in range(size):
        acum += probs[i]

    for i in range(size):
        probs[i] /= acum

    return probs

def useSensor(pos):
    dist = getGhostDistance(pos, ghost)
    table = model[dist]
    colorIdx = selectRandom(table)
    return translation[colorIdx]

def getGhostDistance(pos, ghost):
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

def revealGhost():
    print(ghost)
