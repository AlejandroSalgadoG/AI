from __future__ import division

import random

from Distribution import *

def rand():
    return random.randint(0,100) / 100

def getEvidenceVar(var, evidence):
    evidenceDesc = [desc[1:] for desc in evidence]
    for idx, desc in enumerate(evidenceDesc):
        if desc == var:
            return evidence[idx]
    return None

def getElem(var, dist):
    for desc, prob in dist:
        if desc == var:
            return (desc, prob)

def selectElem(table):
    gen = rand()
    for desc, prob in table:
        gen -= prob
        if gen <= 0:
            return (desc, prob)

def probC(evidence):
    evidenceVar = getEvidenceVar("c", evidence)

    if evidenceVar is None:
        return selectElem(cloudy)
    else:
        return getElem(evidenceVar, cloudy)

def probR(evidence, probC):
    evidenceVar = getEvidenceVar("r", evidence)

    desc, prob = probC
    table = rain[desc]

    if evidenceVar is None:
        return selectElem(table)
    else:
        return getElem(evidenceVar, table)

def probS(evidence, probC):
    evidenceVar = getEvidenceVar("s", evidence)

    desc, prob = probC
    table = sprinkler[desc]

    if evidenceVar is None:
        return selectElem(table)
    else:
        return getElem(evidenceVar, table)

def probW(evidence, probS, probR):
    evidenceVar = getEvidenceVar("w", evidence)

    descS, probS = probS
    descR, probR = probR
    table = wetgrass[descS][descR]

    if evidenceVar is None:
        return selectElem(table)
    else:
        return getElem(evidenceVar, table)
