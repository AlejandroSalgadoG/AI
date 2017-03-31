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

    return table[0]
    #return selectElem(table)

def probW(probS, probR):
    descS, probS = probS
    descR, probR = probR
    table = wetgrass[descS][descR]

    return table[0]
    #return selectElem(table)

def selectElem(table):
    gen = rand()
    for desc, prob in table:
        gen -= prob
        if gen <= 0:
            return (desc, prob)
