import random

from Standars import *
from Model import *

ghost = None

def rand():
    return random.randint(0,100) / 100

def generateGhost():
    global ghost
    i = random.randint(0, numRow-1)
    j = random.randint(0, numRow-1)
    ghost = (i,j)
    print(ghost)

def getInitialDist():
    posNum = numRow**2
    prob = 1 / posNum

    return [prob for i in range(posNum)]

def isGhostThere(pos):
    return pos == ghost

def useSensor(pos):
    dist = getGhostDistance(pos, ghost)

    if dist > maxDist:
        dist = maxDist
    
    table = model[dist]
    colorIdx = selectRandom(table)
    return translation[colorIdx]

def getGhostDistance(pos, ghost):
    i,j = pos
    gi, gj = ghost
    return abs(i-gi) + abs(j-gj)

def getNewDistBase(pos, color, probs):
    newDist = []

    for i in range(numRow):
        for j in range(numRow):
            imgGhost = (i,j)

            pex = calcCondProb(pos, color, imgGhost)
            px = getPosProb(imgGhost, probs)

            newDist.append(pex * px)

    return normalize(newDist)

def getPosProb(pos, probs):
    i,j = pos
    return probs[j+i*numRow]

def getNewDistRec(pos, color, probs):
    newDist = []

    for g in range(numRow):
        for h in range(numRow):
            imgGhost = (g,h)

            acum = 0
            for i in range(numRow):
                for j in range(numRow):
                    xt1Key = (i,j)
                    xt1 = j + i * numRow 

                    xt = fromPosToIdx(imgGhost)

                    table = transition[xt1Key]

                    pxt = table[xt]
                    b = probs[xt1]

                    acum += pxt * b

            pex = calcCondProb(pos, color, imgGhost)

            newDist.append(pex * acum)
        

    return normalize(newDist)

def calcCondProb(pos, color, imgGhost):
    dist = getGhostDistance(pos, imgGhost)

    idx = translation[color]
    return model[dist][idx]

def calcForwardProb(probs):
    newProbs = []

    for i in range(numRow):
        for j in range(numRow):
            xt = (i,j)
            result = 0
            for k in range(numRow):
                for l in range(numRow):
                    xt_1 = (k,l)

                    idx = fromPosToIdx(xt_1)
                    pxt_1 = probs[idx]

                    table = transition(xt_1)

                    idx = fromPosToIdx(xt)
                    pxt = table[idx]

                    result += pxt * pxt_1
            newProbs.append(result)

    return newProbs

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
    table = transition(ghost)
    idx = selectRandom(table)
    ghost = fromIdxToPos(idx)
    print(ghost)

def revealGhost():
    print("The ghost was in", ghost)
