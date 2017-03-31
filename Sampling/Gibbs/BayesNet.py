import random

from Distribution import *

def rand():
    return random.randint(0,100) / 100

def probC():
    return selectElem(cloudy)

def probR(probC):
    desc, prob = probC
    table = rain[desc]

    return selectElem(table)

def probS(probC):
    desc, prob = probC
    table = sprinkler[desc]

    return selectElem(table)

def probW(probS, probR):
    descS, probS = probS
    descR, probR = probR
    table = wetgrass[descS][descR]

    return selectElem(table)

def selectElem(table):
    gen = rand()
    for desc, prob in table:
        gen -= prob
        if gen <= 0:
            return (desc, prob)

def getRandVar(query):
    querySz = len(query)

    idx = random.randint(0,querySz-1)

    return query[idx]

def getCConditioned(desc, samples):
    return [("+c",0.2),("-c", 0.8)]

def getRConditioned(desc, samples):
    return [("+r",0.2),("-r", 0.8)]

def getSConditioned(desc, samples):
    return [("+s",0.2),("-s", 0.8)]

def getWConditioned(desc, samples):
    return [("+w",0.2),("-w", 0.8)]
