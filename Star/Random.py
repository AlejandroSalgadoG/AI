import random

def getRandPos():
    return random.randint(1,25)

def selectRandom(probs):
    randVal = rand()
    size = len(probs)

    for i in range(size):
        randVal -= probs[i]
        if randVal <= 0:
            return i

    return size-1

def rand():
    return random.randint(0,100) / 100
