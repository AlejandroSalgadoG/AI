import random

from Standars import *
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
    print(ghost)

def useSensor(pos):
    dist = getGhostDistance(pos, ghost)
    color = selectRandom(model[dist])
    return translation[color]

def getGhostDistance(pos, ghost):
    i,j = pos
    gi, gj = ghost
    return abs(i-gi) + abs(j-gj)

def distributeParticles(particles, probs):
    part = []
    for i in range(particles):
        idx = selectRandom(probs)
        pos = fromIdxToPos(idx)
        part.append(pos)

    return part

def redistributeParticles(particles, probs):
    part = []
    for i in particles:
        idx = selectRandom(probs)
        particle = particles[idx]
        part.append(particle)

    return part

def weightParticles(particles, condProbs):
    weights = []

    for particle in particles:
        idx = fromPosToIdx(particle)
        prob = condProbs[idx]
        weights.append(prob)

    return weights

def moveParticles(particles):
    newParticles = []

    for particle in particles:
        table = transition(particle)
        idx = selectRandom(table)
        pos = fromIdxToPos(idx)
        newParticles.append(pos)

    return newParticles

def getNewPosDist(pos, color, probs):
    newDist = []

    for i in range(numRow):
        for j in range(numRow):
            imgGhost = (i,j)

            dist = getGhostDistance(pos, imgGhost)

            if dist > maxDist:
                dist = maxDist

            table = model[dist]
            colorIdx = translation[color]
            posIdx = j + i*numRow

            psf = table[colorIdx]
            pf = probs[posIdx]

            newDist.append( psf * pf )

    return normalize(newDist)

def getProbs(particles):
    total = len(particles)
    probs = [0 for i in range(numRow **2)]

    for part in particles:
        pos = fromPosToIdx(part)
        probs[pos] += 1

    for i in range(numRow**2):
        probs[i] /= total

    return probs

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

def isGhostThere(pos):
    return pos == ghost

def revealGhost():
    print("The ghost was in", ghost)
