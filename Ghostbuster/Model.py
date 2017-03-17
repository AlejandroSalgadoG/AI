import random

green = 'G'
yellow = 'Y'
orange = 'O'
red = 'R'

model = []
ghost = None

def rand():
    return random.randint(1,10)

def generateGhost():
    i = random.randint(0,2)
    j = random.randint(0,2)
    return (i,j)

def initializeModel():
    global model,ghost

    ghost = generateGhost()

    #P(col | pos)
                 # 0     1     2    3     4
    model.append([0.4,  0.3,  0.2, 0.15, 0.1 ]) # g
    model.append([0.3,  0.4,  0.5, 0.25, 0.2 ]) # y
    model.append([0.2,  0.2,  0.2, 0.4,  0.25]) # o
    model.append([0.1,  0.1,  0.1, 0.2,  0.45]) # r

def getInitialPosDist():
    posDist = [0 for i in range(5)]

    for i in range(3):
        for j in range(3):
            dist = getGhostDistance(i,j)
            posDist[dist] += 1

    for i in range(5):
        posDist[i] *= 0.11 

    return posDist

def getColor():
    prob = rand()
    if prob <= 3: # 0.3
        return green
    if prob >= 4 and prob <= 5: # 0.2
        return yellow
    if prob >= 6 and prob <= 7: # 0.2
        return orange
    if prob >= 8: # 0.3
        return red

def getModel(color):
    if color == green:
        return model[0]
    elif color == yellow:
        return model[1]
    elif color == orange:
        return model[2]
    else: 
        return model[3]

def updatePosDist(postDist, color):
    newDistr = []
    distr = getModel(color)

    acum = 0
    for i in range(5):
        prob = postDist[i] * distr[i]
        acum += prob
        newDistr.append(prob)    

    for i in range(5):
        newDistr[i] /= acum

    return newDistr

def updatePosMat(pmatrix, posDist):
    for i in range(3):
        for j in range(3):
            distance = getGhostDistance(i,j)
            pmatrix[i][j] = posDist[distance]

    return pmatrix

def getGhostDistance(i,j):
    gi, gj = ghost
    return abs(i-gi) + abs(j-gj)
