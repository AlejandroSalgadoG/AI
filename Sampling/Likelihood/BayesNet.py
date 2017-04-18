from __future__ import division

import random

from Distribution import *

def rand():
    return random.randint(0,100) / 100

def getVarSample(table, evidence):
    if not evidence:
        return selectElem(table)

    for desc, prob in table:
        if desc in evidence:
            return getElem(desc, table)
    return selectElem(table)

def getElem(var, dist):
    for desc, prob in dist:
        if desc == var:
            return (desc, prob)

def selectElem(table):
    randVal = rand()
    for desc, prob in table:
        randVal -= prob
        if randVal <= 0:
            return (desc, prob)
    return table[-1]

def probC(evidence):
    return getVarSample(cloudy, evidence)

def probS(evidence, probC):
    desc, prob = probC
    table = sprinkler[desc]
    return getVarSample(table, evidence)

def probR(evidence, probC):
    desc, prob = probC
    table = rain[desc]
    return getVarSample(table, evidence)

def probW(evidence, probS, probR):
    descS, probS = probS
    descR, probR = probR
    table = wetgrass[descS][descR]
    return getVarSample(table, evidence)
