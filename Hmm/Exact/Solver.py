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

#def getDistVector():
#
#    distVec = { 0 : [],
#                1 : [],
#                2 : [],
#                3 : [],
#                4 : [] }
#
#    for i in range(numRow):
#        for j in range(numRow):
#            pos = (i,j)
#            dist = getGhostDistance(pos)
#            distVec[dist].append(pos)
#
#    return distVec

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
    gi, gj = ghost

    posAvail = getPosAvail()

    if gi == 0:
        if gj == 2:
            probs = getPosProbs(posAvail, "down")
        else:
            probs = getPosProbs(posAvail, "right")
    if gi == 1:
        if gj == 0:
            probs = getPosProbs(posAvail, "up")
        if gj == 1:
            probs = getPosProbs(posAvail, "right")
        if gj == 2:
            probs = getPosProbs(posAvail, "down")
    if gi == 2:
        if gj == 0:
            probs = getPosProbs(posAvail, "up")
        else:
            probs = getPosProbs(posAvail, "left")

    idx = selectRandom(probs)

    ghost = posAvail[idx]

def getPosAvail():
    gi, gj = ghost
    pos = [(gi,gj)]

    if gj+1 <= 2:
        pos.append((gi,gj+1))
    if gi+1 <= 2:
        pos.append((gi+1,gj))
    if gj-1 >= 0:
        pos.append((gi,gj-1))
    if gi-1 >= 0:
        pos.append((gi-1,gj))

    return pos

def getPosProbs(posAvail, direction):
    compProb = 0.5 / (len(posAvail) - 1)
    gi, gj = ghost
    probs = []

    right = (gi,gj+1)
    down = (gi+1,gj)
    left = (gi,gj-1)
    up = (gi-1,gj)

    if direction == "right":
        for pos in posAvail:
            if pos == right:
                probs.append(0.5)
            else:
                probs.append(compProb)
    elif direction == "down":
        for pos in posAvail:
            if pos == down:
                probs.append(0.5)
            else:
                probs.append(compProb)
    elif direction == "left":
        for pos in posAvail:
            if pos == left:
                probs.append(0.5)
            else:
                probs.append(compProb)
    elif direction == "up":
        for pos in posAvail:
            if pos == up:
                probs.append(0.5)
            else:
                probs.append(compProb)
    else:
        print("Error: Unrecognized direction!")

    return probs


